# EE 434 Programming Project

## Project Structure

```bash
/
├── main.py                   # Runs all questions end-to-end
├── src/
│   ├── __init__.py
│   ├── random_generator.py   # Q1 – UniformRandomGenerator
│   ├── distributions.py      # Q2 – ExponentialGenerator, PoissonGenerator
│   ├── mm1_queue.py          # Q3 – MM1Queue 
│   └── mek1_queue.py         # Q4 – MEk1Queue, ErlangGenerator
├── tests/
│   ├── test_q1a.py
│   ├── test_q1b.py
│   ├── test_q2a.py
│   ├── test_q2b.py
│   ├── test_q3a.py
│   ├── test_q3b.py
│   ├── test_q3c.py
│   ├── test_q4a.py
│   ├── test_q4b.py
│   └── test_q4c.py
├── outputs/                  # Generated figures 
└── README.md
```

---

## Requirements

- Python 3.10+  
- `numpy`  
- `matplotlib`

Create enviornment:

```bash
python3 -m venv venv
```

Activate enviornment:

```bash
source venv/bin/activate   # mac

venv\Scripts\activate   # windows command line 

venv\Scripts\Activate.ps1  # windows powershell 
```

Upgrade pip (possible bug fix):

```bash
pip install --upgrade pip
```

Install dependencies:

```bash
pip install numpy matplotlib
```

Or:

```bash
pip install -r requirements.txt
```

When you are done working, deactivate env:
```bash
deactivate
```

---

## Running the Project

Run everything at once:

```bash
python main.py    #May have to use python3 in case of an import error
```

Output figures will be saved to `./outputs/`.

Run an individual question:

```bash
python tests/test_q1a.py
python tests/test_q1b.py
python tests/test_q2a.py
python tests/test_q2b.py
python tests/test_q3a.py
python tests/test_q3b.py
python tests/test_q3c.py
python tests/test_q4a.py
python tests/test_q4b.py
python tests/test_q4c.py
```