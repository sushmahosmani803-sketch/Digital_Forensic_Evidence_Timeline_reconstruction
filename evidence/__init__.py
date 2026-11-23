"""AI-Based Digital Evidence Timeline Reconstruction - core library.

Modules
-------
config     paths and shared constants
normalize  parse the raw CERT insider-threat CSV (two record layouts)
features   time and behavioural feature engineering (leak-free)
models     Random Forest / Isolation Forest training, scoring, evaluation
timeline   chronological reconstruction, sessions, gaps, user risk profiles
report     investigation report generation and robust report loading
"""

__version__ = "2.0.0"
