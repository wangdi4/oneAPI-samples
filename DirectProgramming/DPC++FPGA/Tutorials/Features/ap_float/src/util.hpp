#include <utility>
using DoublePair = std::pair<double, double>;

const inline DoublePair quadratic_gold(double a, double b, double c) {
  auto rooted = b * b - 4.0 * a * c;
  auto rooted_abs = fabs(rooted);

  DoublePair ret;
  if (rooted > 0.0 || rooted_abs < 1e-20) {
    auto root = sqrt(rooted_abs);
    ret = std::make_pair((-b + root) / (2.0 * a), (-b - root) / (2.0 * a));
  } else {
    ret = std::make_pair(NAN, NAN);
  }
  return ret;
}