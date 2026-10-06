"""Public API of the loader package.

Re-exports the three loaders so consumers depend on ``src.loader`` as a
facade instead of on internal modules; moving a function between files
then only changes this file.
"""

from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import BYTE_CATEGORY, Vocab, load_vocab

# Export list for 'from src.loader import *'; named imports work regardless.
__all__ = ["BYTE_CATEGORY", "Vocab", "load_functions", "load_prompts", "load_vocab"]
