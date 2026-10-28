"""
Week 3 Task: Route Optimization Analysis - Logistics
Run: python route_optimization.py
Needs: pandas numpy matplotlib seaborn   (Dijkstra and GA are written from scratch)
Creates: CSV files, 7 charts (charts/ folder), results.json

PART A: Network shortest path   -> Dijkstra (optimal)  vs  Greedy heuristic
PART B: Multi-stop delivery     -> Random vs Nearest Neighbour (greedy) vs 2-opt vs Genetic Algorithm
"""
import os, json, time, heapq, math, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 11
random.seed(SEED); np.random.seed(SEED)
sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1971c2", "#e8590c", "#2f9e44", "#e03131", "#868e96", "#7048e8"
res = {}

# =====================================================================
# PART A: ROAD NETWORK SIMULATION
# =====================================================================
cities = {"Delhi": (28.61, 77.21), "Jaipur": (26.91, 75.79), "Lucknow": (26.85, 80.95), "Ahmedabad": (23.02, 72.57),
          "Mumbai": (19.07, 72.88), "Pune": (18.52, 73.86), "Hyderabad": (17.38, 78.48), "Bengaluru": (12.97, 77.59),
          "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36), "Bhubaneswar": (20.30, 85.82), "Nagpur": (21.15, 79.09)}
names = list(cities)

def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [*cities[a], *cities[b]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))

# connect every city to its 3 nearest neighbours (undirected roads)
pairs = set()
for a in names:
    for b in sorted([x for x in names if x != a], key=lambda x: haversine(a, x))[:3]:
        pairs.add(tuple(sorted((a, b))))
rows = []
for a, b in sorted(pairs):
    km = haversine(a, b) * np.random.uniform(1.15, 1.35)             # road is longer than straight line
    speed = np.random.uniform(45, 65)                                 # average truck speed km/h
    delay = np.random.uniform(0, 1.5) * km / 300                      # traffic / border delay hours
    toll = km * np.random.uniform(0.5, 6.0)                           # toll INR (some roads are toll-heavy)
    rows.append([a, b, round(km, 1), round(km / speed + delay, 2), round(km * 22 + toll)])  # fuel INR 22/km
edges = pd.DataFrame(rows, columns=["from", "to", "distance_km", "time_hrs", "cost_inr"])
edges.to_csv("network_edges.csv", index=False)
pd.DataFrame([[c, *v] for c, v in cities.items()], columns=["city", "lat", "lon"]).to_csv("network_cities.csv", index=False)

def build_graph(df, banned_edges=(), banned_nodes=()):
    g = {c: {} for c in names if c not in banned_nodes}
    for r in df.itertuples():
        if r.to in banned_nodes or r._1 in banned_nodes: continue
        if (r._1, r.to) in banned_edges or (r.to, r._1) in banned_edges: continue
        w = {"distance_km": r.distance_km, "time_hrs": r.time_hrs, "cost_inr": r.cost_inr}
        g[r._1][r.to] = w; g[r.to][r._1] = w
    return g

def dijkstra(g, src, dst, key):
    dist, prev, pq = {src: 0}, {}, [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst: break
        if d > dist.get(u, 1e18): continue
        for v, w in g[u].items():
            nd = d + w[key]
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in dist: return None, None
    path = [dst]
    while path[-1] != src: path.append(prev[path[-1]])
    return path[::-1], dist[dst]

def greedy_path(g, src, dst, key):
    """Always move to the unvisited neighbour that is closest (straight line) to the destination; backtrack at dead ends."""
    path, visited = [src], {src}
    while path[-1] != dst:
        cur = path[-1]
        opts = [v for v in g[cur] if v not in visited]
        if not opts:
            path.pop()
            if not path: return None, None
            continue
        nxt = min(opts, key=lambda v: haversine(v, dst))
        visited.add(nxt); path.append(nxt)
    return path, sum(g[a][b][key] for a, b in zip(path, path[1:]))

G = build_graph(edges)
res["network"] = {"cities": len(names), "roads": len(edges)}

# ---- A1: all origin-destination pairs, 3 objectives, Dijkstra vs Greedy
recs = []
for key in ["distance_km", "time_hrs", "cost_inr"]:
    for o in names:
        for d in names:
            if o == d: continue
            p1, v1 = dijkstra(G, o, d, key); p2, v2 = greedy_path(G, o, d, key)
            recs.append([key, o, d, v1, v2, (v2 - v1) / v1 * 100, len(p1) - 1, len(p2) - 1, p1 == p2])
cmp = pd.DataFrame(recs, columns=["objective", "origin", "destination", "dijkstra", "greedy", "gap_pct", "hops_dij", "hops_greedy", "same_path"])
cmp.to_csv("network_od_comparison.csv", index=False)
res["A_pairs"] = int(len(cmp) / 3)
res["A_gap"] = cmp.groupby("objective")["gap_pct"].agg(["mean", "max"]).round(1).to_dict("index")
res["A_same_path_pct"] = (cmp.groupby("objective")["same_path"].mean() * 100).round(1).to_dict()
res["A_greedy_worse_pct"] = (cmp.groupby("objective")["gap_pct"].apply(lambda s: (s > 0.01).mean()) * 100).round(1).to_dict()
worst = cmp.sort_values("gap_pct").iloc[-1]
res["A_worst"] = [worst.objective, worst.origin, worst.destination, round(worst.gap_pct, 1)]

# ---- A2: scenario analysis - effect of the objective (distance / time / cost)
sc = []
for key in ["distance_km", "time_hrs", "cost_inr"]:
    tot = np.zeros(3); n = 0
    for o in names:
        for d in names:
            if o == d: continue
            p, _ = dijkstra(G, o, d, key)
            tot += [sum(G[a][b][k] for a, b in zip(p, p[1:])) for k in ["distance_km", "time_hrs", "cost_inr"]]; n += 1
    sc.append([key, *(tot / n)])
sc = pd.DataFrame(sc, columns=["optimised_for", "avg_km", "avg_hrs", "avg_cost"]).round(1)
res["A_scenarios"] = sc.to_dict("records")
sc.to_csv("network_objective_scenarios.csv", index=False)

# example OD pair
EX = ("Delhi", "Chennai")
res["A_example"] = {}
for key in ["distance_km", "time_hrs", "cost_inr"]:
    p, _ = dijkstra(G, *EX, key)
    res["A_example"][key] = {"path": p, "km": round(sum(G[a][b]["distance_km"] for a, b in zip(p, p[1:])), 1),
                             "hrs": round(sum(G[a][b]["time_hrs"] for a, b in zip(p, p[1:])), 1),
                             "cost": round(sum(G[a][b]["cost_inr"] for a, b in zip(p, p[1:])))}
pg, _ = greedy_path(G, *EX, "distance_km")
res["A_example"]["greedy"] = {"path": pg, "km": round(sum(G[a][b]["distance_km"] for a, b in zip(pg, pg[1:])), 1)}

# ---- A3: disruption scenarios
pbase, _ = dijkstra(G, *EX, "time_hrs")
base_t = res["A_example"]["time_hrs"]["hrs"]
close = max(zip(pbase, pbase[1:]), key=lambda e: G[e[0]][e[1]]["time_hrs"])      # close the slowest road on the path
G2 = build_graph(edges, banned_edges=[close])
p2, v2 = dijkstra(G2, *EX, "time_hrs")
res["A_closure"] = {"closed": list(close), "before": pbase, "after": p2, "before_hrs": base_t, "after_hrs": round(v2, 1),
                    "increase_pct": round((v2 - base_t) / base_t * 100, 1)}
hub = "Nagpur"
G3 = build_graph(edges, banned_nodes=[hub])
inc, lost = [], 0
for o in names:
    for d in names:
        if o == d or hub in (o, d): continue
        _, a = dijkstra(G, o, d, "distance_km"); pth, b = dijkstra(G3, o, d, "distance_km")
        if pth is None: lost += 1
        else: inc.append((b - a) / a * 100)
res["A_hub_failure"] = {"hub": hub, "avg_increase_pct": round(np.mean(inc), 1), "max_increase_pct": round(np.max(inc), 1),
                        "affected_pct": round(np.mean(np.array(inc) > 0.01) * 100, 1), "unreachable": lost}

# ---- A charts
def draw_net(ax, paths=None, title=""):
    for r in edges.itertuples():
        (la1, lo1), (la2, lo2) = cities[r._1], cities[r.to]
        ax.plot([lo1, lo2], [la1, la2], color="#ced4da", lw=1.5, zorder=1)
    for c, (la, lo) in cities.items():
        ax.scatter(lo, la, s=90, color="#212529", zorder=3); ax.annotate(c, (lo, la), xytext=(5, 5), textcoords="offset points", fontsize=8.5, fontweight="bold")
    for p, col, lab, lw in (paths or []):
        xs = [cities[c][1] for c in p]; ys = [cities[c][0] for c in p]
        ax.plot(xs, ys, color=col, lw=lw, zorder=2, label=lab)
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

fig, ax = plt.subplots(figsize=(8.5, 7.5))
draw_net(ax, title="Simulated Logistics Road Network (12 cities, %d roads)" % len(edges))
for r in edges.itertuples():
    (la1, lo1), (la2, lo2) = cities[r._1], cities[r.to]
    ax.annotate(f"{r.distance_km:.0f}", ((lo1 + lo2) / 2, (la1 + la2) / 2), fontsize=7, color=BLUE, ha="center")
plt.tight_layout(); plt.savefig("charts/1_network_map.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8.5, 7.5))
ex = res["A_example"]
draw_net(ax, [(ex["distance_km"]["path"], BLUE, f"Shortest distance ({ex['distance_km']['km']} km)", 5),
              (ex["time_hrs"]["path"], GREEN, f"Fastest ({ex['time_hrs']['hrs']} hrs)", 3),
              (ex["cost_inr"]["path"], ORANGE, f"Cheapest (INR {ex['cost_inr']['cost']})", 1.6)],
         title="Delhi to Chennai: Best Route Depends on the Objective (Dijkstra)")
ax.legend(loc="lower left"); plt.tight_layout(); plt.savefig("charts/2_dijkstra_objectives.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
g = cmp.copy(); g["objective"] = g["objective"].map({"distance_km": "Distance", "time_hrs": "Time", "cost_inr": "Cost"})
sns.boxplot(data=g, x="objective", y="gap_pct", hue="objective", palette=[BLUE, GREEN, ORANGE], legend=False, ax=ax[0])
ax[0].set_title("Greedy: Extra Cost Above the Optimal (Dijkstra)"); ax[0].set_ylabel("Gap vs optimal (%)"); ax[0].set_xlabel("")
nm = sc.copy(); base = nm.iloc[0, 1:].astype(float).values
nm2 = nm.set_index("optimised_for")
norm = nm2.div(nm2.loc["distance_km"]) * 100
norm.index = ["Distance", "Time", "Cost"]; norm.columns = ["Avg distance", "Avg time", "Avg cost"]
norm.plot(kind="bar", ax=ax[1], color=[BLUE, GREEN, ORANGE], rot=0)
ax[1].axhline(100, color=GREY, ls="--"); ax[1].set_title("Effect of Objective (distance-optimal = 100)")
ax[1].set_xlabel("Route optimised for"); ax[1].set_ylabel("Index"); ax[1].set_ylim(80, 135)
plt.tight_layout(); plt.savefig("charts/3_greedy_gap_and_objectives.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8.5, 7.5))
cl = res["A_closure"]
draw_net(ax, [(cl["before"], GREEN, f"Normal route ({cl['before_hrs']} hrs)", 5), (cl["after"], RED, f"After closure ({cl['after_hrs']} hrs)", 2.5)],
         title=f"Disruption: Road {cl['closed'][0]}-{cl['closed'][1]} Closed (Delhi to Chennai, fastest route)")
(la1, lo1), (la2, lo2) = cities[cl["closed"][0]], cities[cl["closed"][1]]
ax.plot([lo1, lo2], [la1, la2], color="black", lw=2, ls=":", zorder=4); ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("charts/4_disruption_scenario.png", dpi=130); plt.close()

# =====================================================================
# PART B: MULTI-STOP DELIVERY ROUTING (capacity-limited, single depot)
# =====================================================================
rng = np.random.default_rng(3)
N, CAP, KMH, SERVICE_MIN, COST_KM, TRIP_FIXED = 30, 600, 30, 10, 14, 250
xy = np.vstack([[20, 20], rng.uniform(0, 40, (N, 2))])                  # node 0 = depot (km grid)
demand = np.concatenate([[0], rng.integers(20, 121, N)])               # kg per stop
D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) * 1.3              # road distance matrix (km)
pd.DataFrame({"stop": range(N + 1), "x_km": xy[:, 0].round(1), "y_km": xy[:, 1].round(1), "demand_kg": demand}).to_csv("delivery_stops.csv", index=False)

def split(order):
    """cut a stop sequence into trips so that no trip carries more than CAP kg"""
    trips, cur, load = [], [], 0
    for s in order:
        if load + demand[s] > CAP: trips.append(cur); cur, load = [], 0
        cur.append(s); load += demand[s]
    if cur: trips.append(cur)
    return trips

def trip_km(t): return D[0, t[0]] + sum(D[a, b] for a, b in zip(t, t[1:])) + D[t[-1], 0]
def total_km(order): return sum(trip_km(t) for t in split(order))

def metrics(trips, secs):
    km = sum(trip_km(t) for t in trips)
    return {"trips": len(trips), "km": round(km, 1), "hours": round(km / KMH + N * SERVICE_MIN / 60, 1),
            "cost": round(km * COST_KM + len(trips) * TRIP_FIXED), "seconds": round(secs, 3)}

out = {}
# --- 1) Random baseline (average of 300 random orders)
t0 = time.time(); rl = []
for _ in range(300):
    o = list(rng.permutation(np.arange(1, N + 1))); rl.append(total_km(o))
rand_order = list(np.random.RandomState(1).permutation(np.arange(1, N + 1)))
out["Random"] = (split(rand_order), time.time() - t0); res["B_random_avg_km"] = round(float(np.mean(rl)), 1)

# --- 2) Nearest Neighbour (greedy, capacity aware)
t0 = time.time(); left, trips = set(range(1, N + 1)), []
while left:
    cur, load, t = 0, 0, []
    while True:
        fit = [s for s in left if load + demand[s] <= CAP]
        if not fit: break
        nxt = min(fit, key=lambda s: D[cur, s]); t.append(nxt); left.remove(nxt); load += demand[nxt]; cur = nxt
    trips.append(t)
out["Nearest Neighbour"] = (trips, time.time() - t0)
nn_order = [s for t in trips for s in t]

# --- 3) 2-opt local search starting from Nearest Neighbour order
t0 = time.time(); best = nn_order[:]; bv = total_km(best); improved = True
while improved:
    improved = False
    for i in range(N - 1):
        for j in range(i + 1, N):
            cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
            v = total_km(cand)
            if v < bv - 1e-9: best, bv, improved = cand, v, True
out["NN + 2-opt"] = (split(best), time.time() - t0)

# --- 4) Genetic Algorithm (random start, tournament selection, order crossover, swap + reversal mutation, elitism)
POP, GENS, PM, ELITE = 150, 700, 0.25, 4
def ox(p1, p2):
    a, b = sorted(random.sample(range(N), 2)); child = [None] * N; child[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in child[a:b + 1]]; k = 0
    for i in range(N):
        if child[i] is None: child[i] = fill[k]; k += 1
    return child
def mutate(o):
    o = o[:]
    if random.random() < PM:
        if random.random() < 0.5: i, j = random.sample(range(N), 2); o[i], o[j] = o[j], o[i]
        else: i, j = sorted(random.sample(range(N), 2)); o[i:j + 1] = o[i:j + 1][::-1]
    return o
t0 = time.time()
pop = [list(np.random.permutation(np.arange(1, N + 1)).tolist()) for _ in range(POP)]
fit = [total_km(p) for p in pop]; hist = []
for g in range(GENS):
    idx = np.argsort(fit); pop = [pop[i] for i in idx]; fit = [fit[i] for i in idx]; hist.append(fit[0])
    new = pop[:ELITE]
    while len(new) < POP:
        p1 = min(random.sample(range(POP), 4)); p2 = min(random.sample(range(POP), 4))
        new.append(mutate(ox(pop[p1], pop[p2])))
    pop = new; fit = [total_km(p) for p in pop]
bi = int(np.argmin(fit)); ga_order = pop[bi]
out["Genetic Algorithm"] = (split(ga_order), time.time() - t0)
hist.append(min(fit))
res["B_ga_params"] = {"population": POP, "generations": GENS, "mutation": PM, "elite": ELITE}

summ = pd.DataFrame({k: metrics(*v) for k, v in out.items()}).T
summ["km_vs_random_pct"] = ((summ["km"] - summ.loc["Random", "km"]) / summ.loc["Random", "km"] * 100).round(1)
summ["km_vs_NN_pct"] = ((summ["km"] - summ.loc["Nearest Neighbour", "km"]) / summ.loc["Nearest Neighbour", "km"] * 100).round(1)
summ.to_csv("delivery_method_comparison.csv")
res["B_summary"] = summ.reset_index().rename(columns={"index": "method"}).to_dict("records")
res["B_stops"] = N; res["B_capacity"] = CAP; res["B_total_demand"] = int(demand.sum()); res["B_min_trips"] = int(math.ceil(demand.sum() / CAP))
res["B_ga_start_km"] = round(hist[0], 1); res["B_ga_end_km"] = round(min(hist), 1)
first_conv = next(i for i, v in enumerate(hist) if v <= min(hist) * 1.01); res["B_ga_gen_within_1pct"] = first_conv

# --- B charts
cols = [BLUE, ORANGE, GREEN, RED, PURPLE, "#f08c00", "#0b7285"]
fig, axs = plt.subplots(2, 2, figsize=(12, 11))
for ax, (m, (tr, _)) in zip(axs.ravel(), out.items()):
    for k, t in enumerate(tr):
        pts = np.vstack([xy[0], xy[t], xy[0]]); ax.plot(pts[:, 0], pts[:, 1], color=cols[k % len(cols)], lw=1.6, marker="o", ms=3.5)
    ax.scatter(*xy[0], marker="*", s=260, color="black", zorder=5)
    ax.set_title(f"{m}: {sum(trip_km(t) for t in tr):.0f} km, {len(tr)} trips"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
plt.tight_layout(); plt.savefig("charts/5_delivery_routes_methods.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
order = list(out)
for a, col, t, c in zip(ax, ["km", "cost", "hours"], ["Total distance (km)", "Total cost (INR)", "Total time (hours)"], [BLUE, ORANGE, GREEN]):
    vals = summ.loc[order, col].astype(float); b = a.bar(range(4), vals, color=c, alpha=.85)
    a.set_xticks(range(4)); a.set_xticklabels(["Random", "Nearest\nNeighbour", "NN +\n2-opt", "Genetic\nAlgorithm"], fontsize=9)
    for i, v in enumerate(vals): a.text(i, v, f"{v:.0f}" if col != "hours" else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    a.set_title(t)
plt.tight_layout(); plt.savefig("charts/6_method_comparison.png", dpi=130); plt.close()

plt.figure(figsize=(9, 4.6))
plt.plot(hist, color=PURPLE, lw=2, label="Genetic Algorithm (best so far)")
plt.axhline(summ.loc["Nearest Neighbour", "km"], color=ORANGE, ls="--", label=f"Nearest Neighbour ({summ.loc['Nearest Neighbour', 'km']:.0f} km)")
plt.axhline(summ.loc["NN + 2-opt", "km"], color=GREEN, ls="--", label=f"NN + 2-opt ({summ.loc['NN + 2-opt', 'km']:.0f} km)")
plt.axhline(res["B_random_avg_km"], color=GREY, ls=":", label=f"Random average ({res['B_random_avg_km']:.0f} km)")
plt.xlabel("Generation"); plt.ylabel("Total distance (km)"); plt.title("Genetic Algorithm Convergence"); plt.legend()
plt.tight_layout(); plt.savefig("charts/7_ga_convergence.png", dpi=130); plt.close()

json.dump(res, open("results.json", "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))