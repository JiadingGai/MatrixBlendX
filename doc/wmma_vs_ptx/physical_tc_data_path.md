This diagram visualizes the "Temporal vs. Spatial" separation that dictates why the registers are shattered into that specific Quad-strided pattern.

Note that Warp is the atomic unit of instruction issue and control-flow on H200; lanes (threads) are the per-element participants controlled by the active mask and predication, not independently scheduled entities.

### 1. The Quad-Strided Register File (Left)

This section shows the contents of **Lane 0** (Thread 0's physical slice of the register file).

* **Spatial Adjacency ($R0$ & $R1$):** These two registers represent the same "time" (Phase 1) but different physical locations in the matrix (Row 0 vs. Row 8). They are sent to the Tensor Core simultaneously to saturate its "Top" and "Bottom" math units.
* **Temporal Separation ($R0$ vs. $R2$):** Notice that $R0$ and $R2$ both represent Row 0. However, $R2$ is "Quad-strided" because it belongs to the second half of the K-dimension ($K=8 \dots 15$). It sits further down the register file because it won't be needed until Phase 2.

### 2. The SM Micro-Sequencer (Top Right)

This is the "brain" that manages the 8-cycle execution of the `HMMA.16816` instruction.

* Because the physical Tensor Core is too small to calculate the whole 16x8x16 matrix in one go, the Micro-sequencer "freezes" the instruction and loops.
* **Phase 1 (Cycles 1-4):** The sequencer triggers the read-enable for $R0$ and $R1$ across all 32 lanes.
* **Phase 2 (Cycles 5-8):** It switches the read-enable to $R2$ and $R3$.

### 3. The Physical Data Path & Temporal Switch (Bottom Right)

This is the most critical part of the "Why."

* There is a **Physical Lane** (a wire) that connects Thread 0's registers to the Tensor Core's "Top Row Input."
* Instead of having two separate wires for $R0$ and $R2$, the hardware uses a **Temporal Switch (Multiplexer)**.
* On Cycles 1-4, the switch connects $R0$ to the wire. On Cycles 5-8, it flips and connects $R2$ to that *exact same wire*.

### Why this explains your "Transpose Trick"

When you use `ldmatrix.trans`, the hardware is essentially pre-arranging the data into these "Phase 1" and "Phase 2" buckets. By the time the Micro-sequencer starts ticking through the cycles, the data is already perfectly staged so that the "Temporal Switch" just has to flip back and forth to feed the Tensor Core its required values in the correct order.

### How the Accumulator ($c\_frag$) layout relates to this, specifically how the results from Phase 1 and Phase 2 are summed together into the final FP32 registers?**

<TODO>
