from backend.qi_manager import QuantumInspireManager

from qiskit import QuantumCircuit

def test_login():

    manager = QuantumInspireManager()
    manager.login()


    # if test1.check_login():
    #     print("login in checked correctly!")

    # provider = test1.provider
    # backend = provider.get_backend("QX emulator")
    # submit_testing_circut(test1, backend)

def submit_testing_circut(user, backend):
    # Create a basic Bell State circuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Show a representation of the quantum circuit:
    print(qc)

    # Run the circuit on Quantum Inspire's platform:
    job = user.submit(qc, backend)

    # Print the results.
    result = job.result()
    print(result.get_counts())

if __name__ == "__main__":
    test_login()
