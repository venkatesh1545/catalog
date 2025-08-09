// solution.cpp
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include "json-develop/single_include/nlohmann/json.hpp"


using namespace std;
using boost::multiprecision::cpp_int;
using json = nlohmann::json;

// -------------------- Rational with big integers --------------------
struct Rational {
    cpp_int num;
    cpp_int den; // always > 0

    Rational(): num(0), den(1) {}
    Rational(const cpp_int &n): num(n), den(1) {}
    Rational(const cpp_int &n, const cpp_int &d) {
        num = n; den = d;
        normalize();
    }

    static cpp_int abs_cpp(const cpp_int &x) { return x < 0 ? -x : x; }

    static cpp_int gcd_cpp(cpp_int a, cpp_int b) {
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        while (b != 0) {
            cpp_int r = a % b;
            a = b;
            b = r;
        }
        return a;
    }

    void normalize() {
        if (den == 0) throw runtime_error("Denominator zero");
        if (den < 0) { num = -num; den = -den; }
        cpp_int g = gcd_cpp(abs_cpp(num), den);
        if (g != 0) { num /= g; den /= g; }
    }

    // operators
    Rational operator+(const Rational &o) const {
        return Rational(num * o.den + o.num * den, den * o.den);
    }
    Rational operator-(const Rational &o) const {
        return Rational(num * o.den - o.num * den, den * o.den);
    }
    Rational operator*(const Rational &o) const {
        return Rational(num * o.num, den * o.den);
    }
    Rational operator/(const Rational &o) const {
        if (o.num == 0) throw runtime_error("Divide by zero rational");
        return Rational(num * o.den, den * o.num);
    }
    Rational& operator+=(const Rational &o) { *this = *this + o; return *this; }
    Rational& operator-=(const Rational &o) { *this = *this - o; return *this; }
    Rational& operator*=(const Rational &o) { *this = *this * o; return *this; }
    Rational& operator/=(const Rational &o) { *this = *this / o; return *this; }

    bool isZero() const { return num == 0; }
};

// -------------------- Helpers --------------------
cpp_int decodeBaseValue(int base, const string &value) {
    cpp_int res = 0;
    for (char ch : value) {
        int digit;
        if ('0' <= ch && ch <= '9') digit = ch - '0';
        else if ('a' <= ch && ch <= 'z') digit = ch - 'a' + 10;
        else if ('A' <= ch && ch <= 'Z') digit = ch - 'A' + 10;
        else throw runtime_error("Invalid character in base-value");
        if (digit >= base) throw runtime_error("Digit >= base");
        res = res * base + digit;
    }
    return res;
}

cpp_int pow_cpp(const cpp_int &x, int e) {
    cpp_int res = 1;
    cpp_int base = x;
    int exp = e;
    while (exp > 0) {
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}

// -------------------- Gauss-Jordan on rationals --------------------
vector<Rational> solveLinearSystem(vector<vector<Rational>> A) {
    // A is k x (k+1) augmented matrix
    int n = (int)A.size();
    int m = (int)A[0].size() - 1;
    if (m != n) throw runtime_error("Expected square system n x n");

    int row = 0, col = 0;
    for (; row < n && col < m; ++col) {
        // find pivot
        int sel = -1;
        for (int i = row; i < n; ++i) {
            if (!A[i][col].isZero()) { sel = i; break; }
        }
        if (sel == -1) continue;
        if (sel != row) swap(A[sel], A[row]);

        // normalize pivot row: make pivot 1
        Rational pivot = A[row][col];
        for (int j = col; j <= m; ++j) A[row][j] /= pivot;

        // eliminate other rows
        for (int i = 0; i < n; ++i) {
            if (i == row) continue;
            if (A[i][col].isZero()) continue;
            Rational factor = A[i][col];
            for (int j = col; j <= m; ++j) {
                A[i][j] -= factor * A[row][j];
            }
        }
        ++row;
    }

    // after Gauss-Jordan, solution is A[i][m]
    vector<Rational> sol(m);
    for (int i = 0; i < m; ++i) sol[i] = A[i][m];
    return sol;
}

// -------------------- Main --------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string filename = "input.json";
    if (argc >= 2) filename = argv[1];

    ifstream ifs(filename);
    if (!ifs.is_open()) {
        cerr << "Failed to open " << filename << "\n";
        return 1;
    }

    json j;
    try {
        ifs >> j;
    } catch (...) {
        cerr << "Invalid JSON input file\n";
        return 1;
    }

    int n = j["keys"]["n"].get<int>();
    int k = j["keys"]["k"].get<int>(); // need k points to solve: degree = k-1
    int degree = k - 1;
    if (k <= 0) { cerr << "Invalid k\n"; return 1; }

    // collect points (x,y). keys are string integers (like "1","2","3",...)
    // We'll pick the first k entries encountered (sorted by key numeric order).
    vector<pair<int, cpp_int>> points; points.reserve(n);

    // collect numeric keys from JSON (excluding "keys")
    vector<int> keys_list;
    for (auto it = j.begin(); it != j.end(); ++it) {
        string key = it.key();
        if (key == "keys") continue;
        // try parse key as int
        try {
            int x = stoi(key);
            keys_list.push_back(x);
        } catch (...) { continue; }
    }
    sort(keys_list.begin(), keys_list.end());

    for (int x : keys_list) {
        if ((int)points.size() >= k) break;
        string key = to_string(x);
        int base = stoi(j[key]["base"].get<string>());
        string val = j[key]["value"].get<string>();
        cpp_int y = decodeBaseValue(base, val);
        points.emplace_back(x, y);
    }

    if ((int)points.size() < k) { cerr << "Not enough points in JSON\n"; return 1; }

    // Build augmented Vandermonde: rows = k, columns = degree+1, augmented is y
    // column 0 = x^0 (1), column 1 = x^1, ..., column degree = x^degree
    int dim = k;
    vector<vector<Rational>> A(dim, vector<Rational>(dim + 1));
    for (int i = 0; i < dim; ++i) {
        int xi = points[i].first;
        cpp_int X = xi;
        for (int p = 0; p <= degree; ++p) {
            cpp_int val = pow_cpp(X, p);
            A[i][p] = Rational(val);
        }
        A[i][dim] = Rational(points[i].second); // RHS
    }

    // Solve
    vector<Rational> sol;
    try {
        sol = solveLinearSystem(A);
    } catch (exception &e) {
        cerr << "Solve error: " << e.what() << "\n";
        return 1;
    }

    // a0 = sol[0] is c (constant term)
    Rational c = sol[0];
    // print c in integer if denominator == 1 else as num/den
    if (c.den == 1) {
        cout << c.num << "\n";
    } else {
        cout << c.num << "/" << c.den << "\n";
    }

    return 0;
}
