"""Pilot copies of pipeline stages — match + passage type support.

These are EXPERIMENTAL copies of production stage modules with surgical
modifications to support matchQuestion and to fix passageQuestion schema
bugs. Used for the passage+match pilot upload (see memory:
project_passage_match_pilot.md).

After pilot succeeds end-to-end, the bug fixes (especially passage schema
fixes in stage4) should be merged back into production. Match-type support
should be promoted to production after broader QA.

DO NOT IMPORT FROM PRODUCTION CODE PATHS.
"""
