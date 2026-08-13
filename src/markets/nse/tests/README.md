# NSE domain tests

Domain tests live in the repository-level `tests/` suite so pytest discovery and
coverage remain centralized. Boundary tests assert that these modules never depend
on OANDA or Forex risk/persistence implementations.
