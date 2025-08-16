#include <bits/stdc++.h>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <fstream>

using namespace std;

// ----------------------------- Thread-safe Node -----------------------------
class Node {
    unordered_map<string, string> kv_;
    mutable shared_mutex mtx_;
    int id_;
public:
    explicit Node(int id) : id_(id) {}

    void put(const string& k, const string& v) {
        unique_lock lock(mtx_);
        kv_[k] = v;
    }

    bool get(const string& k, string& out) const {
        shared_lock lock(mtx_);
        auto it = kv_.find(k);
        if (it == kv_.end()) return false;
        out = it->second;
        return true;
    }

    bool erase(const string& k) {
        unique_lock lock(mtx_);
        return kv_.erase(k) > 0;
    }

    // Simple checkpoint (overwrite)
    bool saveToFile(const string& path) const {
        ofstream ofs(path, ios::trunc);
        if (!ofs) return false;
        shared_lock lock(mtx_);
        for (auto& [k, v] : kv_) ofs << k << '\t' << v << '\n';
        return true;
    }

    int id() const { return id_; }
};

// ------------------------ Consistent Hashing Ring ---------------------------
struct Hasher {
    // 64-bit mix for strings
    uint64_t operator()(const string& s) const {
        return std::hash<string>{}(s);
    }
    uint64_t operator()(uint64_t x) const {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33; return x;
    }
};

class ConsistentHashRing {
    // map from point -> nodeIndex
    map<uint64_t, int> ring_;
    int vnodes_;
    Hasher H;

    static string vnodeKey(int nodeIdx, int v) {
        return "node#" + to_string(nodeIdx) + "-v" + to_string(v);
    }

public:
    explicit ConsistentHashRing(int virtualNodes = 64) : vnodes_(virtualNodes) {}

    void addNode(int nodeIdx) {
        for (int v = 0; v < vnodes_; ++v) {
            uint64_t h = H(vnodeKey(nodeIdx, v));
            ring_[h] = nodeIdx;
        }
    }

    void removeNode(int nodeIdx) {
        for (int v = 0; v < vnodes_; ++v) {
            uint64_t h = H(vnodeKey(nodeIdx, v));
            auto it = ring_.find(h);
            if (it != ring_.end() && it->second == nodeIdx) ring_.erase(it);
        }
    }

    // Return a sequence of node indices (successors on the ring), possibly with duplicates; caller should uniquify and check liveness.
    vector<int> successors(const string& key, int want) const {
        vector<int> out;
        if (ring_.empty()) return out;
        uint64_t h = H(key);
        auto it = ring_.lower_bound(h);
        for (int i = 0; i < want; ++i) {
            if (it == ring_.end()) it = ring_.begin();
            out.push_back(it->second);
            ++it;
        }
        return out;
    }
};

// ------------------------------- Cluster ------------------------------------
class Cluster {
    vector<shared_ptr<Node>> nodes_;
    vector<bool> alive_;
    ConsistentHashRing ring_;
    int rf_; // replication factor
public:
    Cluster(int nNodes, int replicationFactor, int virtualNodes = 64)
        : alive_(nNodes, true), ring_(virtualNodes), rf_(replicationFactor)
    {
        if (nNodes <= 0) throw runtime_error("Need at least 1 node");
        if (rf_ <= 0) rf_ = 1;
        rf_ = min(rf_, nNodes);
        nodes_.reserve(nNodes);
        for (int i = 0; i < nNodes; ++i) {
            nodes_.push_back(make_shared<Node>(i));
            ring_.addNode(i);
        }
    }

    void setNodeAlive(int idx, bool up) {
        if (idx < 0 || idx >= (int)nodes_.size()) return;
        alive_[idx] = up;
        // We keep points in ring static; we just skip dead nodes during routing.
    }

    vector<int> pickReplicas(const string& key) const {
        // gather many successors, then filter unique & alive
        vector<int> cand = ring_.successors(key, (int)nodes_.size() * 2);
        vector<int> picked;
        unordered_set<int> seen;
        for (int idx : cand) {
            if ((int)picked.size() >= rf_) break;
            if (!alive_[idx]) continue;
            if (seen.insert(idx).second) picked.push_back(idx);
        }
        // In worst case (too many dead), picked could be < rf_. That's okay.
        return picked;
    }

    void put(const string& k, const string& v) {
        auto reps = pickReplicas(k);
        for (int idx : reps) nodes_[idx]->put(k, v);
    }

    bool get(const string& k, string& out) const {
        auto reps = pickReplicas(k);
        for (int idx : reps) {
            if (nodes_[idx]->get(k, out)) return true;
        }
        return false;
    }

    bool erase(const string& k) {
        bool ok = false;
        auto reps = pickReplicas(k);
        for (int idx : reps) ok = nodes_[idx]->erase(k) || ok;
        return ok;
    }

    bool checkpointAll(const string& dir = "chkpt") const {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        bool ok = true;
        for (auto& n : nodes_) {
            string path = dir + "/node_" + to_string(n->id()) + ".tsv";
            ok = n->saveToFile(path) && ok;
        }
        return ok;
    }

    int size() const { return (int)nodes_.size(); }
};

// ------------------------------ Demo / Test ---------------------------------
struct WorkloadConfig {
    int numThreads = 8;
    int opsPerThread = 50000;
    int keySpace = 5000;
    int getBiasPct = 60;  // % of GETs
    int putBiasPct = 35;  // % of PUTs
    // remaining % = DELETEs
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Cluster: 6 nodes, replication factor 3, 64 virtual nodes
    Cluster cluster(6, 3, 64);

    // Optional: background checkpoint thread every 2 seconds
    atomic<bool> stopCheckpoint{false};
    thread chk([&](){
        while (!stopCheckpoint.load()) {
            this_thread::sleep_for(chrono::seconds(2));
            cluster.checkpointAll("chkpt");
        }
    });

    WorkloadConfig cfg;
    cout << "Starting workload with " << cfg.numThreads << " threads...\n";

    atomic<long long> puts{0}, gets{0}, dels{0}, hits{0}, misses{0};

    auto worker = [&](int tid){
        std::mt19937_64 rng(1234567ULL + tid);
        std::uniform_int_distribution<int> keyDist(0, cfg.keySpace - 1);
        std::uniform_int_distribution<int> opDist(1, 100);

        for (int i = 0; i < cfg.opsPerThread; ++i) {
            int k = keyDist(rng);
            string key = "k" + to_string(k);
            int r = opDist(rng);

            if (r <= cfg.getBiasPct) { // GET
                string out;
                if (cluster.get(key, out)) {
                    hits++;
                } else {
                    misses++;
                }
                gets++;
            } else if (r <= cfg.getBiasPct + cfg.putBiasPct) { // PUT
                string val = "v" + to_string(tid) + "_" + to_string(i);
                cluster.put(key, val);
                puts++;
            } else { // DELETE
                cluster.erase(key);
                dels++;
            }

            // Simulate occasional node failures/recoveries
            if (i % 20000 == 10000 && tid == 0) {
                int victim = (i / 20000) % cluster.size();
                cluster.setNodeAlive(victim, false);
                // brief outage
                this_thread::sleep_for(chrono::milliseconds(50));
                cluster.setNodeAlive(victim, true);
            }
        }
    };

    vector<thread> threads;
    threads.reserve(cfg.numThreads);
    auto t0 = chrono::steady_clock::now();
    for (int t = 0; t < cfg.numThreads; ++t) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();
    auto t1 = chrono::steady_clock::now();

    stopCheckpoint.store(true);
    chk.join();

    double secs = chrono::duration<double>(t1 - t0).count();
    long long totalOps = puts + gets + dels;

    cout << "---- Results ----\n";
    cout << "Ops total: " << totalOps << " in " << fixed << setprecision(3) << secs << "s\n";
    cout << "Throughput: " << (long long)(totalOps / max(secs, 1e-9)) << " ops/sec\n";
    cout << "PUT: " << puts << ", GET: " << gets << ", DEL: " << dels << "\n";
    cout << "GET hits: " << hits << ", misses: " << misses << "\n";
    cout << "Checkpoint files written to ./chkpt/\n";

    // One manual checkpoint at the end
    cluster.checkpointAll("chkpt_final");
    cout << "Final checkpoint saved to ./chkpt_final/\n";
    return 0;
}
