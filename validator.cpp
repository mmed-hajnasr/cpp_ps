#include "testlib.h"
#include <cassert>
using namespace std;

#define ll long long
const ll max_val = 50'000'000'000;
const ll max_testcases = 1'000'000;

int main(int argc, char **argv) {
  registerValidation(argc, argv);
  int n = inf.readInt(1, max_testcases, "number of testcases");
  // ll R, t, x, y, v1, v2;
  inf.readEoln();
  for (ll i = 0; i < n; i++) {
    ll R = inf.readLong(0, max_val);
    inf.readSpace();
    ll t = inf.readLong(0, max_val);
    inf.readSpace();
    ll x = inf.readLong(0, max_val);
    inf.readSpace();
    ll y = inf.readLong(0, max_val);
    inf.readSpace();
    ll v1 = inf.readLong(0, max_val);
    inf.readSpace();
    ll v2 = inf.readLong(0, max_val);
    assert(R >= sqrt(pow(x, 2) + pow(y, 2)));
    inf.readEoln();
  }
  inf.readEof();
  return 0;
}
