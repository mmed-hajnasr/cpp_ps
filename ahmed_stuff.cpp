// template
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
#define FAST ios::sync_with_stdio(0), cin.tie(0), cout.tie(0)
#define ll long long
#define ld long double
#define int long long
#define endl "\n"
#define yes cout << "YES" << endl;
#define no cout << "NO" << endl;
#define pb push_back
// #pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
using namespace std;
const int MOD = 1e9 + 7;
// const int MOD = 998244353  ;
const int N = 1e5 + 5;
const ll INF = 1e18;
const ll MIN = -1e18;
typedef tree<ll, null_type, less<ll>, rb_tree_tag,
             tree_order_statistics_node_update>
    indexed_set;

void solve() {}

signed main() {
  FAST;
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ll t = 1;
  cin >> t;
  while (t--)
    solve();
}

// segment tree iterative
template <class T> struct Seg { // comb(ID,b) = b
  const T ID = 0;
  ll n;
  vector<T> seg;
  T comb(T a, T b) { return min(a, b); }
  void init(ll _n) {
    n = _n;
    seg.assign(2*n, ID);
  }
  void pull(ll p) { seg[p] = comb(seg[2*p], seg[2 * p + 1]); }
  void upd(ll p, T val) { // set val at position p
    seg[p += n] = val;
    for (p /= 2; p; p /= 2)
      pull(p);
  }
  T query(ll l, ll r) { // sum on llerval [l, r]
    T ra = ID, rb = ID;
    for (l += n, r += n + 1; l < r; l /= 2, r /= 2) {
      if (l & 1)
        ra = comb(ra, seg[l++]);
      if (r & 1)
        rb = comb(seg[--r], rb);
    }
    return comb(ra, rb);
  }
};

// segment tree recursive
const int N_max = 2e5 + 48;

int Tree[4 * N], a[N];
int n;

int comb(int x, int y) { return __gcd(x, y); }

void build(int ns = 1, int ne = n, int id = 0) {
  if (ns == ne) {
    Tree[id] = a[ns];
    return;
  }
  int idl = 2 * id + 1, idr = 2 * id + 2, md = ns + (ne - ns) / 2;
  build(ns, md, idl);
  build(md + 1, ne, idr);
  Tree[id] = comb(Tree[idl], Tree[idr]);
}

int query(int qs, int qe, int ns = 1, int ne = n, int id = 0) {
  if (ne < qs || ns > qe)
    return INF;
  if (qs <= ns && ne <= qe)
    return Tree[id];
  int idl = 2 * id + 1, idr = 2 * id + 2, md = ns + (ne - ns) / 2;
  int one = query(qs, qe, ns, md, idl);
  int two = query(qs, qe, md + 1, ne, idr);
  return comb(one, two);
}

void update(int pos, int x, int ns = 1, int ne = n, int id = 0) {
  if (pos < ns || pos > ne)
    return;
  if (ns == ne) {
    Tree[id] = x;
    return;
  }
  int idl = 2 * id + 1, idr = 2 * id + 2, md = ns + (ne - ns) / 2;
  update(pos, x, ns, md, idl);
  update(pos, x, md + 1, ne, idr);
  Tree[id] = comb(Tree[idl], Tree[idr]);
}

// lazy segment tree
int Tree[4 * N], a[N];
int lazy[4 * N], n;
int upd[4 * N];

int merge(int x, int y) { return x & y; }

void unlazy(int id, int ns, int ne) {
  if (upd[id] == 0)
    return;
  int l = 2 * id + 1, r = 2 * id + 2;
  Tree[id] = lazy[id];
  lazy[l] = lazy[r] = lazy[id];
  upd[id] = 0;
  upd[l] = upd[r] = 1;
}

void build(int ns = 1, int ne = n, int id = 0) {
  if (ns == ne) {
    Tree[id] = a[ns];
    return;
  }
  int l = 2 * id + 1;
  int r = 2 * id + 2;
  int md = ns + (ne - ns) / 2;
  build(ns, md, l);
  build(md + 1, ne, r);
  Tree[id] = merge(Tree[l], Tree[r]);
}

void update(int qs, int qe, int val, int ns = 1, int ne = n, int id = 0) {
  unlazy(id, ns, ne);
  if (qs > ne || qe < ns)
    return;
  if (qs <= ns && ne <= qe) {
    lazy[id] = val;
    upd[id] = 1;
    unlazy(id, ns, ne);
    return;
  }
  int l = 2 * id + 1, r = 2 * id + 2, md = ns + (ne - ns) / 2;
  update(qs, qe, val, ns, md, l);
  update(qs, qe, val, md + 1, ne, r);
  Tree[id] = merge(Tree[l], Tree[r]);
}

int query(int qs, int qe, int ns = 1, int ne = n, int id = 0) {
  unlazy(id, ns, ne);
  if (qs > ne || qe < ns)
    return (1 << 30) - 1;
  if (qs <= ns && ne <= qe) {
    return Tree[id];
  }
  int l = 2 * id + 1, r = 2 * id + 2, md = ns + (ne - ns) / 2;
  return merge(query(qs, qe, ns, md, l), query(qs, qe, md + 1, ne, r));
}

// dynamic segment tree
int Tree[30 * N];
int l_child[30 * N];
int r_child[30 * N];
int cnt;

int get_val(int id) {
  if (id == -1)
    return 0;
  return Tree[id];
}
int creat() {
  Tree[cnt] = 0;
  l_child[cnt] = r_child[cnt] = -1;
  return cnt++;
}

int query(int qs, int qe, int ns = 1, int ne = 1e9, int id = 0) {
  if (id == -1)
    return 0;
  if (ne < qs || ns > qe)
    return 0;
  if (qs <= ns && ne <= qe)
    return Tree[id];
  int idl = l_child[id], idr = r_child[id], md = ns + (ne - ns) / 2;
  int one = query(qs, qe, ns, md, idl);
  int two = query(qs, qe, md + 1, ne, idr);
  return one + two;
}

int update(int pos, int x, int ns = 1, int ne = 1e9, int id = 0) {
  if (pos < ns || pos > ne)
    return id;
  if (id == -1)
    id = creat();
  if (ns == ne) {
    Tree[id] += x;
    return id;
  }
  int md = ns + (ne - ns) / 2;
  int idl = l_child[id];
  int idr = r_child[id];
  l_child[id] = update(pos, x, ns, md, idl);
  r_child[id] = update(pos, x, md + 1, ne, idr);
  Tree[id] = get_val(l_child[id]) + get_val(r_child[id]);
  return id;
}

// lazy dynamic segment tree
int Tree[32 * N];
int l_child[32 * N];
int r_child[32 * N];
int lazy[32 * N];
int upd[32 * N];
int cnt;

int get_val(int id) {
  if (id == -1)
    return 0;
  return Tree[id];
}

int creat() {
  Tree[cnt] = 0;
  lazy[cnt] = 0;
  upd[cnt] = 0;
  l_child[cnt] = r_child[cnt] = -1;
  return cnt++;
}

void unlazy(int id, int ns, int ne) {
  if (id == -1)
    return;
  if (upd[id] == 0)
    return;
  if (l_child[id] == -1) {
    l_child[id] = creat();
  }
  if (r_child[id] == -1) {
    r_child[id] = creat();
  }
  Tree[id] = (ne - ns + 1);
  lazy[l_child[id]] = lazy[r_child[id]] = lazy[id];
  upd[id] = 0;
  upd[r_child[id]] = upd[l_child[id]] = 1;
}

int query(int qs, int qe, int ns = 1, int ne = 1e9, int id = 0) {
  if (id == -1)
    return 0;
  unlazy(id, ns, ne);
  if (ne < qs || ns > qe)
    return 0;
  if (qs <= ns && ne <= qe)
    return Tree[id];
  int idl = l_child[id], idr = r_child[id], md = ns + (ne - ns) / 2;
  int one = query(qs, qe, ns, md, idl);
  int two = query(qs, qe, md + 1, ne, idr);
  return one + two;
}

int update(int qs, int qe, int val, int ns = 1, int ne = 1e9, int id = 0) {
  unlazy(id, ns, ne);
  if (qs > ne || qe < ns)
    return id;
  if (id == -1)
    id = creat();
  if (qs <= ns && ne <= qe) {
    lazy[id] = val;
    upd[id] = 1;
    unlazy(id, ns, ne);
    return id;
  }
  int md = ns + (ne - ns) / 2;
  int idl = l_child[id];
  int idr = r_child[id];
  l_child[id] = update(qs, qe, val, ns, md, idl);
  r_child[id] = update(qs, qe, val, md + 1, ne, idr);
  Tree[id] = get_val(l_child[id]) + get_val(r_child[id]);
  return id;
}

// MO Algorithm
struct Query {
  int id, l, r;
};

Query qry[N_Max];
int ans[N_Max];
int a[N_Max];
int N, Q;
int l, r;

void add(int val) {}

void rmv(int val) {}

void update(int id) {
  while (r < qry[id].r)
    add(++r);

  while (l > qry[id].l)
    add(--l);

  while (r > qry[id].r)
    rmv(r--);

  while (l < qry[id].l)
    rmv(l++);
}

void mo() {
  int B = sqrt(2 * N);
  sort(qry + 1, qry + Q + 1, [B](Query a, Query b) {
    return make_pair(a.l / B, a.r) < make_pair(b.l / B, b.r);
  });
  l = 1, r = 0;
  for (int i = 1; i <= Q; i++) {
    update(i);
    ans[qry[i].id] = 0;
  }
}

// bipartet matching
struct Hopcroft_Karp {
  // 1-indexing, v is a left node, u is a right node
  int n, m;                      // number of left and right nodes
  vector<basic_string<int>> adj; // adj[v] : right nodes connected to v
  vector<bool> matched;          // matched[v] : whether v is matched
  basic_string<int> left;  // left[u] : the left node which u is matched to
  basic_string<int> depth; // depth[v] : depth of node v in the layered network
  basic_string<int>
      iter; // iter[v] : the iterator over the edges going out of v
  int max_match;

  Hopcroft_Karp() {}
  Hopcroft_Karp(int _n, int _m) { init(_n, _m); }

  void init(int _n, int _m) {
    this->n = _n;
    this->m = _m;
    adj.assign(n + 1, {});
  }

  void add_edge(int v, int u) {
    assert(v && v < n + 1);
    assert(u && u < m + 1);
    adj[v].push_back(u);
  }

  bool bfs() {
    depth.assign(n + 1, -1);
    queue<int> q;
    for (int v = 1; v < n + 1; v++) {
      if (!matched[v]) {
        depth[v] = 0;
        q.push(v);
      }
    }
    bool has_path = 0;
    while (!q.empty()) {
      int v = q.front();
      q.pop();
      for (int u : adj[v]) {
        if (left[u] == -1) {
          has_path = 1;
        } else if (depth[left[u]] == -1) {
          depth[left[u]] = depth[v] + 1;
          q.push(left[u]);
        }
      }
    }
    return has_path;
  }

  bool dfs(int v) {
    for (int &i = iter[v]; i < adj[v].size();) {
      int u = adj[v][i++];
      if (left[u] == -1 || (depth[left[u]] == depth[v] + 1 && dfs(left[u]))) {
        left[u] = v;
        matched[v] = 1;
        return 1;
      }
    }
    depth[v] = -1;
    return 0;
  }

  void build() {
    matched.assign(n + 1, 0);
    left.assign(m + 1, -1);
    max_match = 0;
    while (bfs()) {
      iter.assign(n + 1, 0);
      for (int v = 1; v < n + 1; v++) {
        max_match += (!matched[v] && dfs(v));
      }
    }
  }

  vector<bool> vis_l, vis_r;

  void dfs2(int v) {
    vis_l[v] = 1;
    for (int u : adj[v]) {
      if (!vis_r[u] && left[u] != v) {
        vis_r[u] = 1;
        if (left[u] != -1 && !vis_l[left[u]]) {
          dfs2(left[u]);
        }
      }
    }
  }

  vector<int> min_vertex_cover() {
    vis_l.assign(n, 0);
    vis_r.assign(m, 0);
    for (int v = 1; v < n + 1; v++) {
      if (!matched[v] && !vis_l[v])
        dfs2(v);
    }
    vector<int> res;
    for (int v = 1; v < n + 1; v++) {
      if (!vis_l[v])
        res.push_back(v);
    }
    for (int u = 1; u < m + 1; u++) {
      if (vis_r[u])
        res.push_back(u + n);
    }
    return res;
  }
};

// bipartet matching min cost
const int N = 509;

/* Complexity: O(n^3) but optimized
It finds minimum cost maximum matching.
For finding maximum cost maximum matching
add -cost and return -matching()
1-indexed */
struct Hungarian {
  long long c[N][N], fx[N], fy[N], d[N];
  int l[N], r[N], arg[N], trace[N];
  queue<int> q;
  int start, finish, n;
  const long long inf = 1e18;
  Hungarian() {}
  Hungarian(int n1, int n2) : n(max(n1, n2)) {
    for (int i = 1; i <= n; ++i) {
      fy[i] = l[i] = r[i] = 0;
      for (int j = 1; j <= n; ++j)
        c[i][j] = inf; // make it 0 for maximum cost matching (not necessarily
                       // with max count of matching)
    }
  }
  void add_edge(int u, int v, long long cost) { c[u][v] = min(c[u][v], cost); }
  inline long long getC(int u, int v) { return c[u][v] - fx[u] - fy[v]; }
  void initBFS() {
    while (!q.empty())
      q.pop();
    q.push(start);
    for (int i = 0; i <= n; ++i)
      trace[i] = 0;
    for (int v = 1; v <= n; ++v) {
      d[v] = getC(start, v);
      arg[v] = start;
    }
    finish = 0;
  }
  void findAugPath() {
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int v = 1; v <= n; ++v)
        if (!trace[v]) {
          long long w = getC(u, v);
          if (!w) {
            trace[v] = u;
            if (!r[v]) {
              finish = v;
              return;
            }
            q.push(r[v]);
          }
          if (d[v] > w) {
            d[v] = w;
            arg[v] = u;
          }
        }
    }
  }
  void subX_addY() {
    long long delta = inf;
    for (int v = 1; v <= n; ++v)
      if (trace[v] == 0 && d[v] < delta) {
        delta = d[v];
      }
    // Rotate
    fx[start] += delta;
    for (int v = 1; v <= n; ++v)
      if (trace[v]) {
        int u = r[v];
        fy[v] -= delta;
        fx[u] += delta;
      } else
        d[v] -= delta;
    for (int v = 1; v <= n; ++v)
      if (!trace[v] && !d[v]) {
        trace[v] = arg[v];
        if (!r[v]) {
          finish = v;
          return;
        }
        q.push(r[v]);
      }
  }
  void Enlarge() {
    do {
      int u = trace[finish];
      int nxt = l[u];
      l[u] = finish;
      r[finish] = u;
      finish = nxt;
    } while (finish);
  }
  long long maximum_matching() {
    for (int u = 1; u <= n; ++u) {
      fx[u] = c[u][1];
      for (int v = 1; v <= n; ++v) {
        fx[u] = min(fx[u], c[u][v]);
      }
    }
    for (int v = 1; v <= n; ++v) {
      fy[v] = c[1][v] - fx[1];
      for (int u = 1; u <= n; ++u) {
        fy[v] = min(fy[v], c[u][v] - fx[u]);
      }
    }
    for (int u = 1; u <= n; ++u) {
      start = u;
      initBFS();
      while (!finish) {
        findAugPath();
        if (!finish)
          subX_addY();
      }
      Enlarge();
    }
    long long ans = 0;
    for (int i = 1; i <= n; ++i) {
      if (c[i][l[i]] != inf)
        ans += c[i][l[i]];
      else
        l[i] = 0;
    }
    return ans;
  }
};

// dsu
struct DSU {
  vector<int> par, rnk, sz;
  int c;
  DSU(int n) : par(n + 1), rnk(n + 1, 0), sz(n + 1, 1), c(n) {
    for (int i = 1; i <= n; ++i)
      par[i] = i;
  }
  int find(int i) { return (par[i] == i ? i : (par[i] = find(par[i]))); }
  bool same(int i, int j) { return find(i) == find(j); }
  int get_size(int i) { return sz[find(i)]; }
  int count() {
    return c; // connected components
  }
  int merge(int i, int j) {
    if ((i = find(i)) == (j = find(j)))
      return -1;
    else
      --c;
    if (rnk[i] > rnk[j])
      swap(i, j);
    par[i] = j;
    sz[j] += sz[i];
    if (rnk[i] == rnk[j])
      rnk[j]++;
    return j;
  }
};

// dsu firas edition
struct DSU {
  vector<int> sz, parent;
  void init(int n) {
    for (int i = 0; i < n; i++) {
      parent.pb(i);
      sz.pb(1);
    }
  }
  int get(int u) { return (parent[u] == u ? u : parent[u] = get(parent[u])); }
  bool unite(int u, int v) {
    u = get(u);
    v = get(v);
    if (u == v)
      return false;
    if (sz[u] < sz[v])
      swap(u, v);
    parent[v] = u;
    sz[u] += sz[v];
    return true;
  }
  bool same_set(int u, int v) { return get(u) == get(v); }
  int size(int u) { return sz[get(u)]; }
};

// hld
template <class T> struct Seg {
  const T ID = 0;
  ll n;
  vector<T> seg;
  T comb(T a, T b) { return a + b; }
  void init(ll _n) {
    n = _n;
    seg.assign(2 * n, ID);
  }
  void pull(ll p) { seg[p] = comb(seg[2 * p], seg[2 * p + 1]); }
  void upd(ll p, T val) {
    seg[p += n] = val;
    for (p /= 2; p; p /= 2)
      pull(p);
  }
  T query(ll l, ll r) {
    T ra = ID, rb = ID;
    for (l += n, r += n + 1; l < r; l /= 2, r /= 2) {
      if (l & 1)
        ra = comb(ra, seg[l++]);
      if (r & 1)
        rb = comb(seg[--r], rb);
    }
    return comb(ra, rb);
  }
};

vector<pair<int, int>> adj[N_Max];
int up[N_Max][LOG], sz[N_Max], depth[N_Max];
int a[N_Max], tour[N_Max], top[N_Max], tin[N_Max];
int N, Q, timer = 1;
Seg<int> st;

void dfs_init(int Node, int par) {
  up[Node][0] = par;
  for (int i = 1; i < LOG; i++) {
    up[Node][i] = up[up[Node][i - 1]][i - 1];
  }
  sz[Node] = 1;
  for (auto [child, w] : adj[Node]) {
    if (child == par)
      continue;
    depth[child] = depth[Node] + 1;
    a[child] = w;
    dfs_init(child, Node);
    sz[Node] += sz[child];
  }
}

void HLD(int Node, int tp) {
  int big = -1;
  st.upd(timer, a[Node]);
  tour[timer] = a[Node];
  tin[Node] = timer++;
  top[Node] = tp;
  for (auto [child, w] : adj[Node]) {
    if (child == up[Node][0])
      continue;
    if (big == -1 || sz[child] > sz[big])
      big = child;
  }
  if (big == -1)
    return;
  HLD(big, tp);
  for (auto [child, w] : adj[Node]) {
    if (child != up[Node][0] && child != big)
      HLD(child, child);
  }
}

int get_lca(int u, int v) {
  if (depth[u] > depth[v])
    swap(u, v);
  int l = depth[v] - depth[u];
  for (int i = LOG - 1; i >= 0; i--) {
    if (l & (1 << i))
      v = up[v][i];
  }
  for (int i = LOG - 1; i >= 0; i--) {
    if (up[u][i] != up[v][i]) {
      u = up[u][i];
      v = up[v][i];
    }
  }
  return (u == v ? u : up[u][0]);
}

ll path(int u, int p) {
  ll ret = 0;
  while (u != p) {
    if (top[u] == u) {
      ret += st.query(tin[u], tin[u]);
      u = up[u][0];
    } else if (depth[p] < depth[top[u]]) {
      ret += st.query(tin[top[u]], tin[u]);
      u = up[top[u]][0];
    } else {
      ret += st.query(tin[p] + 1, tin[u]);
      break;
    }
  }
  return ret;
}

// lca
const int N = 3e5 + 9, LG = 18;

vector<int> g[N];
int par[N][LG + 1], dep[N], sz[N];
void dfs(int u, int p = 0) {
  par[u][0] = p;
  dep[u] = dep[p] + 1;
  sz[u] = 1;
  for (int i = 1; i <= LG; i++)
    par[u][i] = par[par[u][i - 1]][i - 1];
  for (auto v : g[u])
    if (v != p) {
      dfs(v, u);
      sz[u] += sz[v];
    }
}
int lca(int u, int v) {
  if (dep[u] < dep[v])
    swap(u, v);
  for (int k = LG; k >= 0; k--)
    if (dep[par[u][k]] >= dep[v])
      u = par[u][k];
  if (u == v)
    return u;
  for (int k = LG; k >= 0; k--)
    if (par[u][k] != par[v][k])
      u = par[u][k], v = par[v][k];
  return par[u][0];
}
int kth(int u, int k) {
  assert(k >= 0);
  for (int i = 0; i <= LG; i++)
    if (k & (1 << i))
      u = par[u][i];
  return u;
}
int dist(int u, int v) {
  int l = lca(u, v);
  if (l != u && l != v)
    return dep[u] + dep[v] - (dep[l] << 1);
  else
    return abs(dep[u] - dep[v]);
}
// kth node from u to v, 0th node is u
int go(int u, int v, int k) {
  int l = lca(u, v);
  int d = dep[u] + dep[v] - (dep[l] << 1);
  assert(k <= d);
  if (dep[l] + k <= dep[u])
    return kth(u, k);
  k -= dep[u] - dep[l];
  return kth(v, dep[v] - dep[l] - k);
}

// matrix
int add(ll a, ll b) { return (a % mod + b % mod) % mod; }

int sub(ll a, ll b) { return (a % mod - b % mod + mod) % mod; }

int mult(ll a, ll b) { return (a % mod) * (b % mod) % mod; }

int divide(ll a, ll b) { return (a % mod) * inv(b) % mod; }

const int N_mat = 105;

struct Matrix {
  int mat[N_mat][N_mat];
  int len;

  Matrix(int n) {
    len = n;
    for (int i = 1; i <= len; i++)
      for (int j = 1; j <= len; j++)
        mat[i][j] = 0;
  }

  Matrix operator*(Matrix other) {
    Matrix ret(len);
    for (int i = 1; i <= len; i++)
      for (int j = 1; j <= len; j++)
        for (int k = 1; k <= len; k++)
          ret.mat[i][j] = add(ret.mat[i][j], mult(mat[i][k], other.mat[k][j]));
    return ret;
  }

  Matrix operator+(Matrix other) {
    Matrix ret(len);
    for (int i = 1; i <= len; i++)
      for (int j = 1; j <= len; j++)
        ret.mat[i][j] = add(mat[i][j], other.mat[i][j]);
    return ret;
  }
};

void print(Matrix M) {
  for (int i = 1; i <= M.len; i++) {
    for (int j = 1; j <= M.len; j++)
      cout << M.mat[i][j] << " ";
    cout << endl;
  }
  cout << endl;
}

Matrix expo_power(Matrix M, ll K) {
  Matrix ret(M.len);
  for (int i = 1; i <= M.len; i++)
    ret.mat[i][i] = 1;
  while (K) {
    if (K & 1)
      ret = ret * M;
    M = M * M;
    K >>= 1;
  }
  return ret;
}

// max_flow1
const int V = 205; // number of vertices
const int E = 405; // 2 * number of edges

#define neig(u, v, e, adj)                                                     \
  for (int e = adj.head[u], v; (e != -1) && (v = adj.to[e], 1); e = adj.nxt[e])

struct ADJ {
  int head[V], to[E], nxt[E], edgecnt;
  ll cap[E];
  int N;

  void init(int n) {
    N = n;
    edgecnt = 0;
    memset(head, -1, n * sizeof(head[0]));
  }

  void addEdge(int u, int v, int c) {
    nxt[edgecnt] = head[u];
    cap[edgecnt] = c;
    to[edgecnt] = v;
    head[u] = edgecnt++;
  }

  void addAugEdge(int u, int v, int c) {
    addEdge(u, v, c);
    addEdge(v, u, 0);
  }
} adj;

struct Dinic { // O(V^2 * E)
  int tmp[E];
  int snk, src, vid;
  vector<int> vis, dist;

  void init(int _src, int _snk) {
    src = _src;
    snk = _snk;
  }

  bool bfs(ADJ &adj) {
    fill(dist.begin(), dist.end(), 0);
    queue<int> q;
    q.push(src);
    vis[src] = vid;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      neig(u, v, e, adj) {
        if (vis[v] == vid || !adj.cap[e])
          continue;
        dist[v] = dist[u] + 1;
        vis[v] = vid;
        q.push(v);
      }
    }
    return vis[snk] == vid;
  }

  ll dfs(int u, ll f, ADJ &adj) {
    if (u == snk)
      return f;
    for (int &e = tmp[u], v; (e != -1) && (v = adj.to[e], 1); e = adj.nxt[e]) {
      if (!adj.cap[e] || dist[v] != dist[u] + 1)
        continue;
      ll new_f = dfs(v, min(f, adj.cap[e]), adj);
      if (new_f) {
        adj.cap[e] -= new_f;
        adj.cap[e ^ 1] += new_f;
        return new_f;
      }
    }
    return 0;
  }

  ll get_flow(ADJ &adj) {
    dist.resize(adj.N);
    vis.assign(adj.N, 0);
    ll maxFlow = 0, flow = 0;
    vid = 1;
    while (bfs(adj)) {
      memcpy(tmp, adj.head, adj.N * sizeof(tmp[0]));
      while (flow = dfs(src, 1e18, adj))
        maxFlow += flow;
      vid++;
    }
    return maxFlow;
  }
} fl;

// max flow2
const int V = 205; // number of vertices
const int E = 405; // 2 * number of edges

#define neig(u, v, e, adj)                                                     \
  for (int e = adj.head[u], v; (e != -1) && (v = adj.to[e], 1); e = adj.nxt[e])

struct ADJ {
  int head[V], to[E], nxt[E], edgecnt;
  ll cap[E];
  int N;

  void init(int n) {
    N = n;
    edgecnt = 0;
    memset(head, -1, n * sizeof(head[0]));
  }

  void addEdge(int u, int v, int c) {
    nxt[edgecnt] = head[u];
    cap[edgecnt] = c;
    to[edgecnt] = v;
    head[u] = edgecnt++;
  }

  void addAugEdge(int u, int v, int c) {
    addEdge(u, v, c);
    addEdge(v, u, 0);
  }
} adj;

struct EdmondsKarp { // O(E^2 * V)
  vector<int> vis, par;
  vector<ll> flow;
  int src, snk, vid;

  void init(int _src, int _snk) {
    src = _src;
    snk = _snk;
  }

  bool bfs(ADJ &adj) {
    fill(flow.begin(), flow.end(), 0);
    queue<int> q;
    q.push(src);
    vis[src] = vid;
    flow[src] = 1e18;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      neig(u, v, e, adj) {
        if (vis[v] == vid || !adj.cap[e])
          continue;
        flow[v] = min(flow[u], adj.cap[e]);
        vis[v] = vid;
        par[v] = e;
        q.push(v);
      }
    }
    return vis[snk] == vid;
  }

  ll get_flow(ADJ &adj) {
    vis.assign(adj.N, 0);
    par.assign(adj.N, 0);
    flow.resize(adj.N);
    ll maxFlow = 0;
    vid = 1;
    while (bfs(adj)) {
      maxFlow += flow[snk];
      int u = snk;
      while (u != src) {
        int e = par[u];
        u = adj.to[e ^ 1];
        adj.cap[e] -= flow[snk];
        adj.cap[e ^ 1] += flow[snk];
      }
      vid++;
    }
    return maxFlow;
  }
} fl;

// max flow3
const int V = 205; // number of vertices
const int E = 405; // 2 * number of edges

#define neig(u, v, e, adj)                                                     \
  for (int e = adj.head[u], v; (e != -1) && (v = adj.to[e], 1); e = adj.nxt[e])

struct ADJ {
  int head[V], to[E], nxt[E], edgecnt;
  ll cap[E];
  int N;

  void init(int n) {
    N = n;
    edgecnt = 0;
    memset(head, -1, n * sizeof(head[0]));
  }

  void addEdge(int u, int v, int c) {
    nxt[edgecnt] = head[u];
    cap[edgecnt] = c;
    to[edgecnt] = v;
    head[u] = edgecnt++;
  }

  void addAugEdge(int u, int v, int c) {
    addEdge(u, v, c);
    addEdge(v, u, 0);
  }
} adj;

struct FlowScaling { // O(E^2 * log(maxCapacity))
  vector<int> vis;
  int src, snk, vid;

  void init(int _src, int _snk) {
    src = _src;
    snk = _snk;
  }

  bool dfs(int u, int f, ADJ &adj) {
    if (u == snk)
      return true;
    vis[u] = vid;
    neig(u, v, e, adj) {
      if (vis[v] == vid || adj.cap[e] < f)
        continue;
      if (dfs(v, f, adj)) {
        adj.cap[e] -= f;
        adj.cap[e ^ 1] += f;
        return true;
      }
    }
    return false;
  }

  ll get_flow(ADJ &adj) {
    vis.assign(adj.N, 0);
    ll maxFlow = 0;
    vid = 1;
    for (int i = 30; i >= 0; i--) {
      while (dfs(src, (1 << i), adj)) {
        maxFlow += (1 << i);
        vid++;
      }
      vid++;
    }
    return maxFlow;
  }
} fl;

// max flow 4
const int V = 205; // number of vertices
const int E = 405; // 2 * number of edges

#define neig(u, v, e, adj)                                                     \
  for (int e = adj.head[u], v; (e != -1) && (v = adj.to[e], 1); e = adj.nxt[e])

struct ADJ {
  int head[V], to[E], nxt[E], edgecnt;
  ll cap[E];
  int N;

  void init(int n) {
    N = n;
    edgecnt = 0;
    memset(head, -1, n * sizeof(head[0]));
  }

  void addEdge(int u, int v, int c) {
    nxt[edgecnt] = head[u];
    cap[edgecnt] = c;
    to[edgecnt] = v;
    head[u] = edgecnt++;
  }

  void addAugEdge(int u, int v, int c) {
    addEdge(u, v, c);
    addEdge(v, u, 0);
  }
} adj;

struct FordFulkerson { // O(maxFlow * (E + V))
  vector<int> vis;
  int src, snk, vid;

  void init(int _src, int _snk) {
    src = _src;
    snk = _snk;
  }

  ll dfs(int u, ll f, ADJ &adj) {
    if (u == snk)
      return f;
    vis[u] = vid;
    neig(u, v, e, adj) {
      if (vis[v] == vid || !adj.cap[e])
        continue;
      ll new_f = dfs(v, min(f, adj.cap[e]), adj);
      if (new_f) {
        adj.cap[e] -= new_f;
        adj.cap[e ^ 1] += new_f;
        return new_f;
      }
    }
    return 0;
  }

  ll get_flow(ADJ &adj) {
    ll maxFlow = 0, flow = 0;
    vis.assign(adj.N, 0);
    vid = 1;
    while (flow = dfs(src, 1e18, adj)) {
      maxFlow += flow;
      vid++;
    }
    return maxFlow;
  }
} fl;

// max flow min cost
const int V = 205; // number of vertices
const int E = 405; // 2 * number of edges

#define neig(u, v, e, adj)                                                     \
  for (int e = adj.head[u], v; (e != -1) && (v = adj.to[e], 1); e = adj.nxt[e])

struct ADJ {
  int head[V], to[E], nxt[E], edgecnt;
  ll cap[E], weight[E];
  int N;

  void init(int n) {
    N = n;
    edgecnt = 0;
    memset(head, -1, n * sizeof(head[0]));
  }

  void addEdge(int u, int v, int c, int w) {
    nxt[edgecnt] = head[u];
    weight[edgecnt] = w;
    cap[edgecnt] = c;
    to[edgecnt] = v;
    head[u] = edgecnt++;
  }

  void addAugEdge(int u, int v, int c, int w) {
    addEdge(u, v, c, w);
    addEdge(v, u, 0, -w);
  }
} adj;

struct MinCostMaxFlow { // O((E * V)^2)
  vector<int> vis, par;
  vector<ll> dist, flow;
  int src, snk, vid;

  void init(int _src, int _snk) {
    src = _src;
    snk = _snk;
  }

  ll BellmanFord(ADJ &adj) {
    fill(dist.begin(), dist.end(), 1e18);
    fill(flow.begin(), flow.end(), 0);
    queue<int> q;
    q.push(src);
    vis[src] = vid;
    dist[src] = 0;
    flow[src] = 1e18;
    while (!q.empty()) {
      int sz = (int)q.size();
      while (sz--) {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        neig(u, v, e, adj) {
          if (!adj.cap[e])
            continue;
          if (dist[u] + adj.weight[e] < dist[v]) {
            dist[v] = dist[u] + adj.weight[e];
            flow[v] = min(flow[u], adj.cap[e]);
            par[v] = e;
            if (vis[v] != vid) {
              vis[v] = vid;
              q.push(v);
            }
          }
        }
      }
    }
    return flow[snk];
  }

  pair<ll, ll> get_flow(ADJ &adj) {
    vis.assign(adj.N, 0);
    par.assign(adj.N, 0);
    dist.resize(adj.N);
    flow.resize(adj.N);
    ll maxFlow = 0, cost = 0, new_flow = 0;
    vid = 1;
    while (new_flow = BellmanFord(adj)) {
      cost += new_flow * dist[snk];
      maxFlow += new_flow;
      int u = snk;
      while (u != src) {
        int e = par[u];
        u = adj.to[e ^ 1];
        adj.cap[e] -= new_flow;
        adj.cap[e ^ 1] += new_flow;
      }
      vid++;
    }
    return {maxFlow, cost};
  }
} fl;

// scc
vector<bool> visited; // keeps track of which vertices are already visited

// runs depth first search starting at vertex v.
// each visited vertex is appended to the output vector when dfs leaves it.
void dfs(int v, vector<vector<int>> const &adj, vector<int> &output) {
  visited[v] = true;
  for (auto u : adj[v])
    if (!visited[u])
      dfs(u, adj, output);
  output.push_back(v);
}

// input: adj -- adjacency list of G
// output: components -- the strongy connected components in G
// output: adj_cond -- adjacency list of G^SCC (by root vertices)
void strongy_connected_components(vector<vector<int>> const &adj,
                                  vector<vector<int>> &components,
                                  vector<vector<int>> &adj_cond) {
  int n = adj.size();
  components.clear(), adj_cond.clear();

  vector<int> order; // will be a sorted list of G's vertices by exit time

  visited.assign(n, false);

  // first series of depth first searches
  for (int i = 1; i < n; i++)
    if (!visited[i])
      dfs(i, adj, order);

  // create adjacency list of G^T
  vector<vector<int>> adj_rev(n);
  for (int v = 1; v < n; v++)
    for (int u : adj[v])
      adj_rev[u].push_back(v);

  visited.assign(n, false);
  reverse(order.begin(), order.end());

  vector<int> roots(n, 0); // gives the root vertex of a vertex's SCC

  // second series of depth first searches
  for (auto v : order)
    if (!visited[v]) {
      std::vector<int> component;
      dfs(v, adj_rev, component);
      sort(component.begin(), component.end());
      components.push_back(component);
      int root = component.front();
      for (auto u : component)
        roots[u] = root;
    }

  // add edges to condensation graph
  adj_cond.assign(n, {});
  for (int v = 1; v < n; v++)
    for (auto u : adj[v])
      if (roots[v] != roots[u])
        adj_cond[roots[v]].push_back(roots[u]);
}

// topological sort

vector<ll> tps;
vector<ll> vis(n + 1, true);
function<void(int)> dfs = [&](int i) {
  vis[i] = false;
  for (auto child : adj[i]) {
    if (vis[child]) {
      dfs(child);
    }
  }
  tps.pb(i);
};
for (int i = 1; i <= n; i++) {
  if (vis[i]) {
    dfs(i);
  }
}
reverse(tps.begin(), tps.end());

// sieve
const int N = 1e6+5  ;
vector<ll> sieve(N,1);
vector<ll> spf(N,INF);
void preproc(){
    for(int i=2;i<N;i++){
        if(sieve[i]==1){
            for(int j=i;j<N;j+=i){
                spf[j]=min(spf[j],i);
                sieve[j]++;
            }
        }
    }
}
 
vector<ll> factors(int n){
    vector<ll> ans;
    while(n!=1){
        ans.pb(spf[n]);
        n/=spf[n];
    }
    return ans;
}
