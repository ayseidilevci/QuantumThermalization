# QuantumThermalization
Quantum thermal state preparation via a quantum computer on the Quantum Inspire cloud

### Installation

1. Clone the repository
```bash
   git clone https://github.com/ayseidilevci/QuantumThermalization.git
```

2. Create a virtual environment
```bash
   python -m venv venv
```

3. Activate the virtual environment
```bash
   # macOS/Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
```

4. Install dependencies
```bash
   pip install -r requirements.txt
```

5. Choose your python interpreter as your virtual environment 
    * For Windows: `Ctrl+Shift+P` --> "Python: Select Interpreter"
    * For Mac: `Cmd+Shift+P` --> "Python: Select Interpreter"


### Software Stack

The implementation relies primarily on:

- Qiskit
- NumPy

Experiments were developed and tested using Python 3.12.