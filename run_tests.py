#!/usr/bin/env python3
"""Test runner with simple CLI flags.

Supported flags:
  --coverage         enable coverage (default)
  --html             produce HTML coverage report in test_reports/coverage_html
  --xml              produce XML coverage report (coverage.xml in test_reports)
  --edge-cases-only  run only tests marked with 'edge'
  --integration-only run only integration tests (marked 'integration')
  --ci               run in CI mode (emits junit xml)

Examples:
  python run_tests.py --html --xml
  python run_tests.py --integration-only --ci
"""
import argparse
import os
import subprocess
import sys


def build_pytest_cmd(args):
    cmd = ["pytest"]

    if args.coverage:
        cmd += ["--cov=.", "--cov-config=.coveragerc"]

    # Coverage reports
    cov_reports = ["term"]
    if args.html:
        cov_reports.append("html:test_reports/coverage_html")
    if args.xml:
        cov_reports.append("xml:test_reports/coverage.xml")

    for r in cov_reports:
        cmd.append(f"--cov-report={r}")

    # Test selection
    if args.edge_only:
        cmd += ["-m", "edge"]
    if args.integration_only:
        cmd += ["-m", "integration"]

    # CI flags
    if args.ci:
        # produce junit xml for CI systems
        os.makedirs('test_reports', exist_ok=True)
        cmd += ["--junitxml=test_reports/junit.xml"]

    return cmd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-coverage', dest='coverage', action='store_false', help='Disable coverage')
    p.add_argument('--coverage', dest='coverage', action='store_true', help='Enable coverage (default)')
    p.set_defaults(coverage=True)
    p.add_argument('--html', action='store_true', help='Produce HTML coverage report')
    p.add_argument('--xml', action='store_true', help='Produce XML coverage report')
    p.add_argument('--edge-cases-only', dest='edge_only', action='store_true', help='Run edge-case tests only')
    p.add_argument('--integration-only', dest='integration_only', action='store_true', help='Run integration tests only')
    p.add_argument('--ci', action='store_true', help='CI mode: produce junit xml in test_reports')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs('test_reports', exist_ok=True)
    cmd = build_pytest_cmd(args)
    print('Running:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == '__main__':
    main()
