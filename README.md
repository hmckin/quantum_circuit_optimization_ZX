# ZXReduce

Build a circuit in Qiskit, hand it to PyZX, run `full_reduce`, extract a new circuit.

## Run it

```
pip install -r requirements.txt
python demo.py
```

`demo.py` takes a ripple-carry adder, transpiles it to Clifford+T, simplifies it
with ZX, and writes the before/after diagrams to `out/`.

`bench.py` runs the same pipeline over a handful of arithmetic circuits and operator oracles.

