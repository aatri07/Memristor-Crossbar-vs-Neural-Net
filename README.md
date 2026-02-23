What even is a memristor?

A memristor is a portmanteau of “memory” and “resistor.” It is a two-terminal passive device whose resistance depends on the history of the voltage or current applied to it; in other words, it remembers its last state even when power is removed. This property allows it to store analog values, unlike traditional MOSFET-based transistors, which store only binary 1s and 0s. As a result, memristors can achieve high-density storage and enable computations directly in memory. 

We learned in CS429 that memory is separate from the processor. Data comes to and from through buses (one of the
reasons why we have a data alignment requirement). In a memristor crossbar, it's not states of flowing current
or not that encodes data, but variable resistance: a continuous spectrum rather than a bistable set. The weights
are stored in the resistance itself, meaning that theoretically there is NO gap between representation and value.
Essentially, a computer with memristors can do computations and store data in the same place! See this diagram of
a memristor crossbar:

         V1     V2     V3
         │      │      │
      ──[G11]─[G12]─[G13]── → I1
         │      │      │
      ──[G21]─[G22]─[G23]── → I2
         │      │      │
      ──[G31]─[G32]─[G33]── → I3

Where I is an output vector of current, V is an input vector in the form of voltage, and G is a conductance matrix where Gij represents each memristor. As per Ohm's law, I = G * V. Therefore, the weights are applied to "bases" in
physics, so the representation is the physical value itself; no conversions like from binary to other bases. The
resistance Rij of each memristor, 1 / Gij, is the actual physical value in memory. It can be changed by some voltage pulses but persists after voltage is turned off. 

"But wait," you ask, "how the hell am I supposed to compute with the memristors without overwriting data when I apply voltage, idiot?" And that's where the beauty of duration comes in. If you want to write a value to a memristor, you apply strong enough voltage for long enough to change the value. But if you simply want to compute, just apply a smaller voltage to elicit the same current. The beauty of memristors is in their non-linearity: below the voltage required to write data, the memristor will behave in this linear way where I = V * G. Above this threshold, the memristor's resistance will change, meaning G won't be constant and I (the vector for current, not myself) won't change linearly. 

Now, memristors sounds amazing, but they have their pitfalls compared to your average MOSFET. The discrete, distinguishable 1 and 0 are resistant against noise in voltage, something memristors can't boast. The logic gates we rely on use these discrete signals (think: discrete math!). Plus, the fact that a certain voltage needs to be applied for a certain amount of time makes memristors slower than their on-off MOSFET counterparts. Until the technology for these devices, which must be calibrated on the nanoscale, improves, memristors are promising, incredibly powerful tools for eliminating the memory-processor gap. Memristors trade precision and speed for parallelism and in-memory computation, making them ideal for AI, analog signal processing, and approximate computing tasks.