from backend.qi_manager import QuantumInspireManager

from qiskit import QuantumCircuit

def test():
    test1 = QuantumInspireManager()
    provider = test1.provider
    backend = 
    if test1.login():
        print("logged in")

def submit_testing_circut():
    # Create a basic Bell State circuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Show a representation of the quantum circuit:
    print(qc)

    # Run the circuit on Quantum Inspire's platform:
    job = test1.submit(qc)

    # Print the results.
    result = job.result()
    print(result.get_counts())
