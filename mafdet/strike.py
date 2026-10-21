"""
mafdet strike: the chat of mafdet watch, with the answer written ahead.

While the user pauses, the whole prompt is already in cache (mafdet watch);
strike also starts generating the answer and keeps it aside. On Enter, if the
message is the one it was written for, the tokens already there appear at once
and the stream goes on live; if the text changed in the meantime, the answer
is thrown away at the change and a new one starts at the next pause.

    mafdet strike
    mafdet strike --ahead 1024      # how many answer tokens to write before Enter
"""
from .watch import run  # noqa: F401  (the same window and session, with --ahead set)
