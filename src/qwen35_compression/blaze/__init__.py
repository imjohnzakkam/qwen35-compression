"""Blaze: spend a checkpoint's bytes where the model uses them.

A 4-bit Qwen3.5 export keeps its 248,320-row vocabulary table in BF16, a third of the 4B's file.
Blaze stores the busy rows at 8 bits and the rest at 4 (the ladder), prices every body unit by
measured KL, and spends the freed bytes on the body. Nothing is trained or Fisher-weighted.
"""
