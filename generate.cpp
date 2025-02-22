#include <bits/stdc++.h>
#include <cstdio>

#include "testlib.h"

using namespace std;

#define ll long long

using namespace std;

int main(int argc, char *argv[]) {

  registerGen(argc, argv, 1);

  ll id = opt<ll>(1);
  ll number_of_tests = opt<ll>(2);
  ll max = opt<ll>(3);
  printf("%lld\n", number_of_tests);

  for (ll i = 0; i < number_of_tests; i++) {
    ll x = rnd.next(max);
    ll y = rnd.next(max);
    ll v1 = rnd.next(max);
    ll v2 = rnd.next(max);
    ll t = rnd.next(max);
    ll R = sqrt(pow(x, 2) + pow(y, 2)) + rnd.next(max) + 1;
    printf("%lld %lld %lld %lld %lld %lld\n", R, t, x, y, v1, v2);
  }
}
