#include <algorithm>
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
#define FAST ios::sync_with_stdio(0), cin.tie(0), cout.tie(0)
#define ll long long
#define ld long double
#define int long long
#define endl "\n"
#define all(x) x.begin(), x.end()
using namespace std;
const int MOD = 1e9 + 7;
const ll INF = 1e18;
const ll MIN = -1e18;
const ll MAX = 1e5;
typedef tree<ll, null_type, less<ll>, rb_tree_tag,
             tree_order_statistics_node_update>
    indexed_set;

const int MAX_BITS = 20;

struct segment {
  int n;
  int basis[MAX_BITS];

  segment() : n(0) {}

  int get_max() const {
    int answer = 0;

    for (int i = 0; i < n; i++)
      answer = max(answer, answer ^ basis[i]);

    return answer;
  }

  void join(const segment &other) {
    if (n == MAX_BITS)
      return;

    for (int i = 0; i < other.n; i++) {
      int x = other.basis[i];

      for (int j = 0; j < n; j++)
        x = min(x, x ^ basis[j]);

      if (x != 0) {
        basis[n++] = x;

        // Insertion sort.
        for (int k = n - 1; k > 0 && basis[k] > basis[k - 1]; k--)
          swap(basis[k], basis[k - 1]);
      }
    }

    assert(n <= MAX_BITS);
  }

  // TODO: decide whether to re-implement this for better performance. Mainly
  // relevant when segments contain arrays.
  void join(const segment &a, const segment &b) {
    *this = a;
    join(b);
  }
};

struct basic_seg_tree {
  int tree_n = 0;
  vector<segment> tree;

  basic_seg_tree(int n = 0) {
    if (n > 0)
      init(n);
  }

  void init(int n) {
    tree_n = n;
    tree.assign(2 * tree_n, segment());
  }

  // O(n) initialization of our tree.
  void build(const vector<segment> &initial) {
    int n = initial.size();
    assert(n <= tree_n);

    for (int i = 0; i < n; i++)
      tree[tree_n + i] = initial[i];

    for (int position = tree_n - 1; position > 0; position--)
      tree[position].join(tree[2 * position], tree[2 * position + 1]);
  }

  segment query(int a, int b) {
    segment answer;

    for (a += tree_n, b += tree_n; a < b; a /= 2, b /= 2) {
      if (a & 1)
        answer.join(tree[a++]);

      if (b & 1)
        answer.join(tree[--b]);
    }

    return answer;
  }
};

void solve() {
  ll N;
  cin >> N;
  vector<segment> initial;

  for (int i = 0; i < N; i++) {
    int burger;
    cin >> burger;
    initial.emplace_back();
    initial.back().n = burger == 0 ? 0 : 1;
    initial.back().basis[0] = burger;
  }

  basic_seg_tree tree(N);
  tree.build(initial);
  ll Q;
  cin >> Q;

  for (int q = 0; q < Q; q++) {
    int L, R;
    cin >> L >> R;
    L--;
    cout << tree.query(L, R).get_max() << endl;
  }
}

signed main() {
  FAST;
  ll t = 1;
  // cin >> t;
  while (t--)
    solve();
}
