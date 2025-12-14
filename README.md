![image](https://github.com/mytechnotalent/HackingGPT-4/blob/main/HackingGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# HackingGPT
## Part 4
Part 4 covers matrix multiplication for efficient token averaging, lower triangular weight matrices, row normalization for averaging, and broadcasting mechanics in PyTorch.

#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)

<br>

## Part 3 [HERE](https://github.com/mytechnotalent/HackingGPT-3)

<br><br>

```python
import torch
```


## Step 1: Load and Inspect the Data
Now let's read the file and see what we're working with. Understanding your data is crucial before building any model!


```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```


```python
text
```


**Output:**
```
'A dim glow rises behind the glass of a screen and the machine exhales in binary tides. The hum is a language and one who listens leans close to catch the quiet grammar. Patterns fold like small maps and seams hint at how the thing holds itself together. Treat each blinking diode and each idle tick as a sentence in a story that asks to be read.\n\nThere is patience here, not of haste but of careful unthreading. Where others see a sealed box the curious hand traces the join and wonders which thought made it fit. Do not rush to break, coax the meaning out with questions, and watch how the logic replies in traces and errors and in the echoes of forgotten interfaces.\n\nTechnology is artifact and argument at once. It makes a claim about what should be simple, what should be hidden, and what should be trusted. Reverse the gaze and learn its rhetoric, see where it promises ease, where it buries complexity, and where it leaves a backdoor as a sigh between bricks. To read that rhetoric is to be a kind interpreter, not a vandal.\n\nThis work is an apprenticeship in humility. Expect bafflement and expect to be corrected by small things, a timing oddity, a mismatch of expectation, a choice that favors speed over grace. Each misstep teaches a vocabulary of trade offs. Each discovery is a map of decisions and not a verdict on worth.\n\nThere is a moral keeping in the craft. Let curiosity be tempered with regard for consequence. Let repair and understanding lead rather than exploitation. The skill that opens a lock should also know when to hold the key and when to hand it back, mindful of harm and mindful of help.\n\nCelebrate the quiet victories, a stubborn protocol understood, an obscure format rendered speakable, a closed device coaxed into cooperation. These are small reconciliations between human intent and metal will, acts of translation rather than acts of conquest.\n\nAfter decoding a mechanism pause and ask what should change, a bug to be fixed, a user to be warned, a design to be amended. The true maker of machines leaves things better for having looked, not simply for having cracked the shell.'
```


## Step 2: Version 2 - Matrix Multiplication (Fast!)
We can do the same averaging as Part 3's for-loops with a single matrix multiplication. The trick is to create a lower-triangular weight matrix.

### Why Matrix Multiplication is Faster
In Part 3, we used nested for-loops to compute averages. This is slow because of the following.
1. Python loops have overhead for each iteration.
2. Each operation happens one at a time (sequential).
3. There is no parallelization or GPU acceleration.

Matrix multiplication is fast because of the following.
1. Operations happen in parallel on the GPU.
2. Highly optimized BLAS (Basic Linear Algebra Subprograms) libraries.
3. Single operation replaces thousands of loop iterations.

### The Key Insight: Lower Triangular Matrices
A lower triangular matrix has zeros above the diagonal. This creates the "only look at past tokens" pattern we need!

| Row | What it "sees" | Pattern |
|-----|----------------|---------|
| 0 | just position 0 | [1, 0, 0, 0, 0, 0, 0, 0] |
| 1 | positions 0, 1 | [1, 1, 0, 0, 0, 0, 0, 0] |
| 2 | positions 0, 1, 2 | [1, 1, 1, 0, 0, 0, 0, 0] |
| 3 | positions 0, 1, 2, 3 | [1, 1, 1, 1, 0, 0, 0, 0] |
| 4 | positions 0, 1, 2, 3, 4 | [1, 1, 1, 1, 1, 0, 0, 0] |
| 5 | positions 0, 1, 2, 3, 4, 5 | [1, 1, 1, 1, 1, 1, 0, 0] |
| 6 | positions 0, 1, 2, 3, 4, 5, 6 | [1, 1, 1, 1, 1, 1, 1, 0] |
| 7 | positions 0, 1, 2, 3, 4, 5, 6, 7 | [1, 1, 1, 1, 1, 1, 1, 1] |

### How Does This Become Averaging?
After we normalize each row to sum to 1.
| Row | Normalized weights | Each weight equals |
|-----|--------------------|--------------------|
| 0 | [1.0, 0, 0, 0, 0, 0, 0, 0] | 1/1 = 1.0 |
| 1 | [0.5, 0.5, 0, 0, 0, 0, 0, 0] | 1/2 = 0.5 |
| 2 | [0.33, 0.33, 0.33, 0, 0, 0, 0, 0] | 1/3 ≈ 0.33 |
| 3 | [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0] | 1/4 = 0.25 |
| 4 | [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0] | 1/5 = 0.2 |
| 5 | [0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0, 0] | 1/6 ≈ 0.167 |
| 6 | [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0] | 1/7 ≈ 0.143 |
| 7 | [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125] | 1/8 = 0.125 |

When we multiply this weight matrix by our input, each output position becomes the weighted sum (which is the same as the average!) of the input positions.


```python
torch.manual_seed(42)
```


**Output:**
```
<torch._C.Generator at 0x103f6a4f0>
```


```python
# define batch dimension
B = 4  # batch size: 4 independent sequences
B
```


**Output:**
```
4
```


```python
# define time dimension
T = 8  # sequence length: 8 tokens/positions in each sequence
T
```


**Output:**
```
8
```


```python
# define channel dimension
C = 2  # feature size: 2 features per token
C
```


**Output:**
```
2
```


```python
# start with random data
x = torch.randn(B, T, C)
x
```


**Output:**
```
tensor([[[ 1.9269,  1.4873],
         [ 0.9007, -2.1055],
         [ 0.6784, -1.2345],
         [-0.0431, -1.6047],
         [-0.7521,  1.6487],
         [-0.3925, -1.4036],
         [-0.7279, -0.5594],
         [-0.7688,  0.7624]],

        [[ 1.6423, -0.1596],
         [-0.4974,  0.4396],
         [-0.7581,  1.0783],
         [ 0.8008,  1.6806],
         [ 1.2791,  1.2964],
         [ 0.6105,  1.3347],
         [-0.2316,  0.0418],
         [-0.2516,  0.8599]],

        [[-1.3847, -0.8712],
         [-0.2234,  1.7174],
         [ 0.3189, -0.4245],
         [ 0.3057, -0.7746],
         [-1.5576,  0.9956],
         [-0.8798, -0.6011],
         [-1.2742,  2.1228],
         [-1.2347, -0.4879]],

        [[-0.9138, -0.6581],
         [ 0.0780,  0.5258],
         [-0.4880,  1.1914],
         [-0.8140, -0.7360],
         [-1.4032,  0.0360],
         [-0.0635,  0.6756],
         [-0.0978,  1.8446],
         [-1.1845,  1.3835]]])
```


```python
# create lower triangular matrix of ones
# torch.ones(T, T) creates an 8x8 matrix filled with 1s
# torch.tril() keeps only the lower triangular part (sets upper to 0)
wei = torch.tril(torch.ones(T, T))
wei
```


**Output:**
```
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])
```


```python
# let's understand what torch.tril does step by step
print('step 1: create a matrix of all ones')
all_ones = torch.ones(T, T)
print(f'torch.ones({T}, {T}) creates:')
print(all_ones)
print()
print('step 2: apply torch.tril to keep only lower triangle')
print('tril = TRIangular Lower')
print('this sets everything ABOVE the diagonal to zero')
print()
print('visual of what gets kept vs zeroed:')
print('row 0: [KEEP, zero, zero, zero, zero, zero, zero, zero]')
print('row 1: [KEEP, KEEP, zero, zero, zero, zero, zero, zero]')
print('row 2: [KEEP, KEEP, KEEP, zero, zero, zero, zero, zero]')
print('row 3: [KEEP, KEEP, KEEP, KEEP, zero, zero, zero, zero]')
print('row 4: [KEEP, KEEP, KEEP, KEEP, KEEP, zero, zero, zero]')
print('row 5: [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, zero, zero]')
print('row 6: [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, zero]')
print('row 7: [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, KEEP]')
print()
print('result:')
print(torch.tril(all_ones))
```


**Output:**
```
step 1: create a matrix of all ones
torch.ones(8, 8) creates:
tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])

step 2: apply torch.tril to keep only lower triangle
tril = TRIangular Lower
this sets everything ABOVE the diagonal to zero

visual of what gets kept vs zeroed:
row 0: [KEEP, zero, zero, zero, zero, zero, zero, zero]
row 1: [KEEP, KEEP, zero, zero, zero, zero, zero, zero]
row 2: [KEEP, KEEP, KEEP, zero, zero, zero, zero, zero]
row 3: [KEEP, KEEP, KEEP, KEEP, zero, zero, zero, zero]
row 4: [KEEP, KEEP, KEEP, KEEP, KEEP, zero, zero, zero]
row 5: [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, zero, zero]
row 6: [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, zero]
row 7: [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, KEEP]

result:
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])

```


```python
# examine each row of wei individually
print('examining each row of the lower triangular matrix:')
print()
print(f'row 0: {wei[0].tolist()}')
print('       position 0 only sees itself')
print('       1 one = will sum 1 position')
print()
print(f'row 1: {wei[1].tolist()}')
print('       position 1 sees positions 0 and 1')
print('       2 ones = will sum 2 positions')
print()
print(f'row 2: {wei[2].tolist()}')
print('       position 2 sees positions 0, 1, and 2')
print('       3 ones = will sum 3 positions')
print()
print(f'row 3: {wei[3].tolist()}')
print('       position 3 sees positions 0, 1, 2, and 3')
print('       4 ones = will sum 4 positions')
print()
print(f'row 4: {wei[4].tolist()}')
print('       position 4 sees positions 0, 1, 2, 3, and 4')
print('       5 ones = will sum 5 positions')
print()
print(f'row 5: {wei[5].tolist()}')
print('       position 5 sees positions 0, 1, 2, 3, 4, and 5')
print('       6 ones = will sum 6 positions')
print()
print(f'row 6: {wei[6].tolist()}')
print('       position 6 sees positions 0, 1, 2, 3, 4, 5, and 6')
print('       7 ones = will sum 7 positions')
print()
print(f'row 7: {wei[7].tolist()}')
print('       position 7 sees all positions 0, 1, 2, 3, 4, 5, 6, and 7')
print('       8 ones = will sum 8 positions')
```


**Output:**
```
examining each row of the lower triangular matrix:

row 0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       position 0 only sees itself
       1 one = will sum 1 position

row 1: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       position 1 sees positions 0 and 1
       2 ones = will sum 2 positions

row 2: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       position 2 sees positions 0, 1, and 2
       3 ones = will sum 3 positions

row 3: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
       position 3 sees positions 0, 1, 2, and 3
       4 ones = will sum 4 positions

row 4: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
       position 4 sees positions 0, 1, 2, 3, and 4
       5 ones = will sum 5 positions

row 5: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
       position 5 sees positions 0, 1, 2, 3, 4, and 5
       6 ones = will sum 6 positions

row 6: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
       position 6 sees positions 0, 1, 2, 3, 4, 5, and 6
       7 ones = will sum 7 positions

row 7: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
       position 7 sees all positions 0, 1, 2, 3, 4, 5, 6, and 7
       8 ones = will sum 8 positions

```


```python
# normalize each row to sum to 1
# wei.sum(dim=1, keepdim=True) sums each row
# dividing makes each row sum to 1 (turning sums into averages)
wei = wei / wei.sum(dim=1, keepdim=True)
wei
```


**Output:**
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```


```python
# let's understand the normalization step by step
print('understanding row normalization:')
print()
print('first, let\'s recreate the lower triangular matrix')
wei_raw = torch.tril(torch.ones(T, T))
print('wei before normalization:')
print(wei_raw)
print()
```


**Output:**
```
understanding row normalization:

first, let's recreate the lower triangular matrix
wei before normalization:
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])


```


```python
# compute the sum of each row
print('step 1: sum each row')
print('wei.sum(dim=1, keepdim=True)')
print()
print('dim=1 means sum along the column axis (sum each row)')
print('keepdim=True keeps the result as a column vector for broadcasting')
print()
row_sums = wei_raw.sum(dim=1, keepdim=True)
print('row sums:')
print(row_sums)
print()
print('explanation:')
print(f'row 0 sum: 1.0 (has 1 one)')
print(f'row 1 sum: 2.0 (has 2 ones)')
print(f'row 2 sum: 3.0 (has 3 ones)')
print(f'row 3 sum: 4.0 (has 4 ones)')
print(f'row 4 sum: 5.0 (has 5 ones)')
print(f'row 5 sum: 6.0 (has 6 ones)')
print(f'row 6 sum: 7.0 (has 7 ones)')
print(f'row 7 sum: 8.0 (has 8 ones)')
```


**Output:**
```
step 1: sum each row
wei.sum(dim=1, keepdim=True)

dim=1 means sum along the column axis (sum each row)
keepdim=True keeps the result as a column vector for broadcasting

row sums:
tensor([[1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.],
        [8.]])

explanation:
row 0 sum: 1.0 (has 1 one)
row 1 sum: 2.0 (has 2 ones)
row 2 sum: 3.0 (has 3 ones)
row 3 sum: 4.0 (has 4 ones)
row 4 sum: 5.0 (has 5 ones)
row 5 sum: 6.0 (has 6 ones)
row 6 sum: 7.0 (has 7 ones)
row 7 sum: 8.0 (has 8 ones)

```


```python
# now divide each row by its sum
print('step 2: divide each row by its sum')
print('wei_normalized = wei_raw / row_sums')
print()
wei_normalized = wei_raw / row_sums
print('normalized wei:')
print(wei_normalized)
print()
print('now each row sums to 1.0!')
```


**Output:**
```
step 2: divide each row by its sum
wei_normalized = wei_raw / row_sums

normalized wei:
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

now each row sums to 1.0!

```


```python
# examine each normalized row in detail
print('examining each normalized row:')
print()
print(f'row 0: {wei[0].tolist()}')
print('       1/1 = 1.0')
print('       position 0 gets 100% weight on itself')
print()
print(f'row 1: {wei[1].tolist()}')
print('       1/2 = 0.5')
print('       position 1 gives 50% weight to position 0, 50% to position 1')
print()
print(f'row 2: {wei[2].tolist()}')
print('       1/3 ≈ 0.333')
print('       position 2 gives 33.3% weight to each of positions 0, 1, 2')
print()
print(f'row 3: {wei[3].tolist()}')
print('       1/4 = 0.25')
print('       position 3 gives 25% weight to each of positions 0, 1, 2, 3')
print()
print(f'row 4: {wei[4].tolist()}')
print('       1/5 = 0.2')
print('       position 4 gives 20% weight to each of positions 0, 1, 2, 3, 4')
print()
print(f'row 5: {wei[5].tolist()}')
print('       1/6 ≈ 0.167')
print('       position 5 gives 16.7% weight to each of positions 0, 1, 2, 3, 4, 5')
print()
print(f'row 6: {wei[6].tolist()}')
print('       1/7 ≈ 0.143')
print('       position 6 gives 14.3% weight to each of positions 0, 1, 2, 3, 4, 5, 6')
print()
print(f'row 7: {wei[7].tolist()}')
print('       1/8 = 0.125')
print('       position 7 gives 12.5% weight to each of positions 0, 1, 2, 3, 4, 5, 6, 7')
```


**Output:**
```
examining each normalized row:

row 0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       1/1 = 1.0
       position 0 gets 100% weight on itself

row 1: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       1/2 = 0.5
       position 1 gives 50% weight to position 0, 50% to position 1

row 2: [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.0, 0.0, 0.0, 0.0, 0.0]
       1/3 ≈ 0.333
       position 2 gives 33.3% weight to each of positions 0, 1, 2

row 3: [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
       1/4 = 0.25
       position 3 gives 25% weight to each of positions 0, 1, 2, 3

row 4: [0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.0, 0.0, 0.0]
       1/5 = 0.2
       position 4 gives 20% weight to each of positions 0, 1, 2, 3, 4

row 5: [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.0, 0.0]
       1/6 ≈ 0.167
       position 5 gives 16.7% weight to each of positions 0, 1, 2, 3, 4, 5

row 6: [0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.0]
       1/7 ≈ 0.143
       position 6 gives 14.3% weight to each of positions 0, 1, 2, 3, 4, 5, 6

row 7: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
       1/8 = 0.125
       position 7 gives 12.5% weight to each of positions 0, 1, 2, 3, 4, 5, 6, 7

```


```python
# verify each row sums to 1
print('verify: each row sums to 1.0')
print()
for i in range(T):
    row_sum = wei[i].sum().item()
    print(f'row {i} sum: {row_sum:.4f}')
```


**Output:**
```
verify: each row sums to 1.0

row 0 sum: 1.0000
row 1 sum: 1.0000
row 2 sum: 1.0000
row 3 sum: 1.0000
row 4 sum: 1.0000
row 5 sum: 1.0000
row 6 sum: 1.0000
row 7 sum: 1.0000

```


```python
# matrix multiply!
# wei @ x performs the weighted averaging
# wei shape: (T, T) = (8, 8)
# x shape: (B, T, C) = (4, 8, 2)
# result shape: (B, T, C) = (4, 8, 2)
x_bow_2 = wei @ x
x_bow_2
```


**Output:**
```
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])
```


### Understanding the Matrix Multiplication Broadcasting
When we do `wei @ x`, PyTorch broadcasts the operation.
- `wei` has shape (T, T) = (8, 8)
- `x` has shape (B, T, C) = (4, 8, 2)

PyTorch treats the batch dimension (B=4) specially. It performs 4 separate matrix multiplications.
- `wei @ x[0]` → result for batch 0
- `wei @ x[1]` → result for batch 1  
- `wei @ x[2]` → result for batch 2
- `wei @ x[3]` → result for batch 3

For each batch, the multiplication is the following.
- (8, 8) @ (8, 2) = (8, 2)

The final result has shape (4, 8, 2) = (B, T, C).


```python
# let's trace through the matrix multiplication step by step for batch 0
print('understanding the matrix multiplication for batch 0')
print()
print(f'wei shape: {wei.shape}')
print(f'x[0] shape: {x[0].shape}')
print()
print('x[0] (the input for batch 0)')
print(x[0])
print()
print('wei (the weight matrix):')
print(wei)
```


**Output:**
```
understanding the matrix multiplication for batch 0

wei shape: torch.Size([8, 8])
x[0] shape: torch.Size([8, 2])

x[0] (the input for batch 0)
tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047],
        [-0.7521,  1.6487],
        [-0.3925, -1.4036],
        [-0.7279, -0.5594],
        [-0.7688,  0.7624]])

wei (the weight matrix):
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

```


```python
# position 0 calculation (row 0 of wei @ x[0])
print('position 0 calculation')
print()
print(f'wei[0] = {wei[0].tolist()}')
print(f'this means: 1.0 * x[0,0] + 0.0 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]')
print()
print('for feature 0')
val = wei[0, 0].item() * x[0, 0, 0].item()
print(f'   {wei[0, 0].item():.4f} * {x[0, 0, 0].item():.4f} = {val:.4f}')
print()
print('for feature 1')
val = wei[0, 0].item() * x[0, 0, 1].item()
print(f'   {wei[0, 0].item():.4f} * {x[0, 0, 1].item():.4f} = {val:.4f}')
print()
print(f'result: x_bow_2[0, 0] = {x_bow_2[0, 0].tolist()}')
print(f'verify: x[0, 0]       = {x[0, 0].tolist()}')
print('(position 0 just equals itself since it only sees itself)')
```


**Output:**
```
position 0 calculation

wei[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
this means: 1.0 * x[0,0] + 0.0 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]

for feature 0
   1.0000 * 1.9269 = 1.9269

for feature 1
   1.0000 * 1.4873 = 1.4873

result: x_bow_2[0, 0] = [1.9269150495529175, 1.4872841835021973]
verify: x[0, 0]       = [1.9269150495529175, 1.4872841835021973]
(position 0 just equals itself since it only sees itself)

```


```python
# position 1 calculation (row 1 of wei @ x[0])
print('position 1 calculation')
print()
print(f'wei[1] = {wei[1].tolist()}')
print(f'this means: 0.5 * x[0,0] + 0.5 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]')
print()
print('for feature 0')
val0 = wei[1, 0].item() * x[0, 0, 0].item()
val1 = wei[1, 1].item() * x[0, 1, 0].item()
print(f'   {wei[1, 0].item():.4f} * {x[0, 0, 0].item():.4f} = {val0:.4f}')
print(f' + {wei[1, 1].item():.4f} * {x[0, 1, 0].item():.4f} = {val1:.4f}')
print(f'   sum = {val0 + val1:.4f}')
print()
print('for feature 1')
val0 = wei[1, 0].item() * x[0, 0, 1].item()
val1 = wei[1, 1].item() * x[0, 1, 1].item()
print(f'   {wei[1, 0].item():.4f} * {x[0, 0, 1].item():.4f} = {val0:.4f}')
print(f' + {wei[1, 1].item():.4f} * {x[0, 1, 1].item():.4f} = {val1:.4f}')
print(f'   sum = {val0 + val1:.4f}')
print()
print(f'result: x_bow_2[0, 1] = {x_bow_2[0, 1].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1]) / 2
print(f'(x[0,0] + x[0,1]) / 2 = {manual_avg.tolist()}')
```


**Output:**
```
position 1 calculation

wei[1] = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
this means: 0.5 * x[0,0] + 0.5 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]

for feature 0
   0.5000 * 1.9269 = 0.9635
 + 0.5000 * 0.9007 = 0.4504
   sum = 1.4138

for feature 1
   0.5000 * 1.4873 = 0.7436
 + 0.5000 * -2.1055 = -1.0528
   sum = -0.3091

result: x_bow_2[0, 1] = [1.4138160943984985, -0.3091186285018921]

manual verification
(x[0,0] + x[0,1]) / 2 = [1.4138160943984985, -0.3091186285018921]

```


```python
# position 2 calculation (row 2 of wei @ x[0])
print('position 2 calculation')
print()
print(f'wei[2] = {wei[2].tolist()}')
print(f'this means: 0.333 * x[0,0] + 0.333 * x[0,1] + 0.333 * x[0,2] + 0.0 * x[0,3] + ...')
print()
print('for feature 0')
val0 = wei[2, 0].item() * x[0, 0, 0].item()
val1 = wei[2, 1].item() * x[0, 1, 0].item()
val2 = wei[2, 2].item() * x[0, 2, 0].item()
print(f'   {wei[2, 0].item():.4f} * {x[0, 0, 0].item():.4f} = {val0:.4f}')
print(f' + {wei[2, 1].item():.4f} * {x[0, 1, 0].item():.4f} = {val1:.4f}')
print(f' + {wei[2, 2].item():.4f} * {x[0, 2, 0].item():.4f} = {val2:.4f}')
print(f'   sum = {val0 + val1 + val2:.4f}')
print()
print('for feature 1')
val0 = wei[2, 0].item() * x[0, 0, 1].item()
val1 = wei[2, 1].item() * x[0, 1, 1].item()
val2 = wei[2, 2].item() * x[0, 2, 1].item()
print(f'   {wei[2, 0].item():.4f} * {x[0, 0, 1].item():.4f} = {val0:.4f}')
print(f' + {wei[2, 1].item():.4f} * {x[0, 1, 1].item():.4f} = {val1:.4f}')
print(f' + {wei[2, 2].item():.4f} * {x[0, 2, 1].item():.4f} = {val2:.4f}')
print(f'   sum = {val0 + val1 + val2:.4f}')
print()
print(f'result: x_bow_2[0, 2] = {x_bow_2[0, 2].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2]) / 3
print(f'(x[0,0] + x[0,1] + x[0,2]) / 3 = {manual_avg.tolist()}')
```


**Output:**
```
position 2 calculation

wei[2] = [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.0, 0.0, 0.0, 0.0, 0.0]
this means: 0.333 * x[0,0] + 0.333 * x[0,1] + 0.333 * x[0,2] + 0.0 * x[0,3] + ...

for feature 0
   0.3333 * 1.9269 = 0.6423
 + 0.3333 * 0.9007 = 0.3002
 + 0.3333 * 0.6784 = 0.2261
   sum = 1.1687

for feature 1
   0.3333 * 1.4873 = 0.4958
 + 0.3333 * -2.1055 = -0.7018
 + 0.3333 * -1.2345 = -0.4115
   sum = -0.6176

result: x_bow_2[0, 2] = [1.168683648109436, -0.6175941228866577]

manual verification
(x[0,0] + x[0,1] + x[0,2]) / 3 = [1.1686835289001465, -0.6175940632820129]

```


```python
# position 3 calculation (row 3 of wei @ x[0])
print('position 3 calculation')
print()
print(f'wei[3] = {wei[3].tolist()}')
print(f'this means: 0.25 * x[0,0] + 0.25 * x[0,1] + 0.25 * x[0,2] + 0.25 * x[0,3] + 0.0 * ...')
print()
print('for feature 0')
vals = [wei[3, i].item() * x[0, i, 0].item() for i in range(4)]
for i in range(4):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[3, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[3, i].item() * x[0, i, 1].item() for i in range(4)]
for i in range(4):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[3, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_2[0, 3] = {x_bow_2[0, 3].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3]) / 4
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3]) / 4 = {manual_avg.tolist()}')
```


**Output:**
```
position 3 calculation

wei[3] = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
this means: 0.25 * x[0,0] + 0.25 * x[0,1] + 0.25 * x[0,2] + 0.25 * x[0,3] + 0.0 * ...

for feature 0
   0.2500 * 1.9269 = 0.4817
 + 0.2500 * 0.9007 = 0.2252
 + 0.2500 * 0.6784 = 0.1696
 + 0.2500 * -0.0431 = -0.0108
   sum = 0.8657

for feature 1
   0.2500 * 1.4873 = 0.3718
 + 0.2500 * -2.1055 = -0.5264
 + 0.2500 * -1.2345 = -0.3086
 + 0.2500 * -1.6047 = -0.4012
   sum = -0.8644

result: x_bow_2[0, 3] = [0.8657457828521729, -0.8643622994422913]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3]) / 4 = [0.8657457828521729, -0.8643622994422913]

```


```python
# position 4 calculation (row 4 of wei @ x[0])
print('position 4 calculation')
print()
print(f'wei[4] = {wei[4].tolist()}')
print(f'this means: 0.2 * x[0,0] + 0.2 * x[0,1] + 0.2 * x[0,2] + 0.2 * x[0,3] + 0.2 * x[0,4] + 0.0 * ...')
print()
print('for feature 0')
vals = [wei[4, i].item() * x[0, i, 0].item() for i in range(5)]
for i in range(5):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[4, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[4, i].item() * x[0, i, 1].item() for i in range(5)]
for i in range(5):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[4, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_2[0, 4] = {x_bow_2[0, 4].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4]) / 5
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4]) / 5 = {manual_avg.tolist()}')
```


**Output:**
```
position 4 calculation

wei[4] = [0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.0, 0.0, 0.0]
this means: 0.2 * x[0,0] + 0.2 * x[0,1] + 0.2 * x[0,2] + 0.2 * x[0,3] + 0.2 * x[0,4] + 0.0 * ...

for feature 0
   0.2000 * 1.9269 = 0.3854
 + 0.2000 * 0.9007 = 0.1801
 + 0.2000 * 0.6784 = 0.1357
 + 0.2000 * -0.0431 = -0.0086
 + 0.2000 * -0.7521 = -0.1504
   sum = 0.5422

for feature 1
   0.2000 * 1.4873 = 0.2975
 + 0.2000 * -2.1055 = -0.4211
 + 0.2000 * -1.2345 = -0.2469
 + 0.2000 * -1.6047 = -0.3209
 + 0.2000 * 1.6487 = 0.3297
   sum = -0.3617

result: x_bow_2[0, 4] = [0.542169451713562, -0.36174529790878296]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4]) / 5 = [0.5421693921089172, -0.36174526810646057]

```


```python
# position 5 calculation (row 5 of wei @ x[0])
print('position 5 calculation')
print()
print(f'wei[5] = {wei[5].tolist()}')
print(f'this means: 0.167 * x[0,0] + 0.167 * x[0,1] + ... + 0.167 * x[0,5] + 0.0 * ...')
print()
print('for feature 0')
vals = [wei[5, i].item() * x[0, i, 0].item() for i in range(6)]
for i in range(6):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[5, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[5, i].item() * x[0, i, 1].item() for i in range(6)]
for i in range(6):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[5, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_2[0, 5] = {x_bow_2[0, 5].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4] + x[0, 5]) / 6
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5]) / 6 = {manual_avg.tolist()}')
```


**Output:**
```
position 5 calculation

wei[5] = [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.0, 0.0]
this means: 0.167 * x[0,0] + 0.167 * x[0,1] + ... + 0.167 * x[0,5] + 0.0 * ...

for feature 0
   0.1667 * 1.9269 = 0.3212
 + 0.1667 * 0.9007 = 0.1501
 + 0.1667 * 0.6784 = 0.1131
 + 0.1667 * -0.0431 = -0.0072
 + 0.1667 * -0.7521 = -0.1254
 + 0.1667 * -0.3925 = -0.0654
   sum = 0.3864

for feature 1
   0.1667 * 1.4873 = 0.2479
 + 0.1667 * -2.1055 = -0.3509
 + 0.1667 * -1.2345 = -0.2058
 + 0.1667 * -1.6047 = -0.2674
 + 0.1667 * 1.6487 = 0.2748
 + 0.1667 * -1.4036 = -0.2339
   sum = -0.5354

result: x_bow_2[0, 5] = [0.38639479875564575, -0.5353888869285583]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5]) / 6 = [0.3863947093486786, -0.5353888869285583]

```


```python
# position 6 calculation (row 6 of wei @ x[0])
print('position 6 calculation')
print()
print(f'wei[6] = {wei[6].tolist()}')
print(f'this means: 0.143 * x[0,0] + 0.143 * x[0,1] + ... + 0.143 * x[0,6] + 0.0 * x[0,7]')
print()
print('for feature 0')
vals = [wei[6, i].item() * x[0, i, 0].item() for i in range(7)]
for i in range(7):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[6, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[6, i].item() * x[0, i, 1].item() for i in range(7)]
for i in range(7):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[6, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_2[0, 6] = {x_bow_2[0, 6].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4] + x[0, 5] + x[0, 6]) / 7
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6]) / 7 = {manual_avg.tolist()}')
```


**Output:**
```
position 6 calculation

wei[6] = [0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.0]
this means: 0.143 * x[0,0] + 0.143 * x[0,1] + ... + 0.143 * x[0,6] + 0.0 * x[0,7]

for feature 0
   0.1429 * 1.9269 = 0.2753
 + 0.1429 * 0.9007 = 0.1287
 + 0.1429 * 0.6784 = 0.0969
 + 0.1429 * -0.0431 = -0.0062
 + 0.1429 * -0.7521 = -0.1074
 + 0.1429 * -0.3925 = -0.0561
 + 0.1429 * -0.7279 = -0.1040
   sum = 0.2272

for feature 1
   0.1429 * 1.4873 = 0.2125
 + 0.1429 * -2.1055 = -0.3008
 + 0.1429 * -1.2345 = -0.1764
 + 0.1429 * -1.6047 = -0.2292
 + 0.1429 * 1.6487 = 0.2355
 + 0.1429 * -1.4036 = -0.2005
 + 0.1429 * -0.5594 = -0.0799
   sum = -0.5388

result: x_bow_2[0, 6] = [0.22721239924430847, -0.5388233065605164]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6]) / 7 = [0.22721242904663086, -0.5388233065605164]

```


```python
# position 7 calculation (row 7 of wei @ x[0])
print('position 7 calculation')
print()
print(f'wei[7] = {wei[7].tolist()}')
print(f'this means: 0.125 * x[0,0] + 0.125 * x[0,1] + ... + 0.125 * x[0,7]')
print()
print('for feature 0')
vals = [wei[7, i].item() * x[0, i, 0].item() for i in range(8)]
for i in range(8):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[7, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[7, i].item() * x[0, i, 1].item() for i in range(8)]
for i in range(8):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[7, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_2[0, 7] = {x_bow_2[0, 7].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4] + x[0, 5] + x[0, 6] + x[0, 7]) / 8
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7]) / 8 = {manual_avg.tolist()}')
```


**Output:**
```
position 7 calculation

wei[7] = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
this means: 0.125 * x[0,0] + 0.125 * x[0,1] + ... + 0.125 * x[0,7]

for feature 0
   0.1250 * 1.9269 = 0.2409
 + 0.1250 * 0.9007 = 0.1126
 + 0.1250 * 0.6784 = 0.0848
 + 0.1250 * -0.0431 = -0.0054
 + 0.1250 * -0.7521 = -0.0940
 + 0.1250 * -0.3925 = -0.0491
 + 0.1250 * -0.7279 = -0.0910
 + 0.1250 * -0.7688 = -0.0961
   sum = 0.1027

for feature 1
   0.1250 * 1.4873 = 0.1859
 + 0.1250 * -2.1055 = -0.2632
 + 0.1250 * -1.2345 = -0.1543
 + 0.1250 * -1.6047 = -0.2006
 + 0.1250 * 1.6487 = 0.2061
 + 0.1250 * -1.4036 = -0.1755
 + 0.1250 * -0.5594 = -0.0699
 + 0.1250 * 0.7624 = 0.0953
   sum = -0.3762

result: x_bow_2[0, 7] = [0.10270600765943527, -0.3761647045612335]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7]) / 8 = [0.10270600765943527, -0.3761647045612335]

```


```python
# print shapes summary
print('version 2: matrix multiplication averaging')
print()
print(f'wei shape:    {wei.shape} → (T={T}, T={T})')
print(f'x shape:      {x.shape} → (B={B}, T={T}, C={C})')
print(f'result shape: {x_bow_2.shape} → (B={B}, T={T}, C={C})')
print()
print('Same output shape as input!')
print('Each position now holds the average of itself and all previous positions.')
```


**Output:**
```
version 2: matrix multiplication averaging

wei shape:    torch.Size([8, 8]) → (T=8, T=8)
x shape:      torch.Size([4, 8, 2]) → (B=4, T=8, C=2)
result shape: torch.Size([4, 8, 2]) → (B=4, T=8, C=2)

Same output shape as input!
Each position now holds the average of itself and all previous positions.

```


### Comparing Version 1 (For-Loops) vs Version 2 (Matrix Multiplication)
Both methods produce the EXACT same result! Let's verify this.


```python
# recreate version 1 result using for-loops (from Part 3)
print('recreating version 1 (for-loop method) for comparison')
print()
x_bow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        x_previous = x[b, :t+1]
        x_bow[b, t] = torch.mean(x_previous, dim=0)
print('x_bow (for-loop result)')
print(x_bow)
```


**Output:**
```
recreating version 1 (for-loop method) for comparison

x_bow (for-loop result)
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])

```


```python
# compare version 1 and version 2 results
print('comparing version 1 (for-loop) vs version 2 (matrix multiplication)')
print()
print('x_bow_2 (matrix multiplication result)')
print(x_bow_2)
print()
print('Are they equal?')
print(f'torch.allclose(x_bow, x_bow_2) = {torch.allclose(x_bow, x_bow_2)}')
print()
print('exact difference (should be all zeros or very close)')
diff = x_bow - x_bow_2
print(f'max absolute difference: {torch.abs(diff).max().item()}')
```


**Output:**
```
comparing version 1 (for-loop) vs version 2 (matrix multiplication)

x_bow_2 (matrix multiplication result)
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])

Are they equal?
torch.allclose(x_bow, x_bow_2) = True

exact difference (should be all zeros or very close)
max absolute difference: 1.1920928955078125e-07

```


```python
# element by element comparison for batch 0
print('element by element comparison for batch 0')
print()
for t in range(T):
    print(f'position {t}')
    print(f'   for-loop result: {x_bow[0, t].tolist()}')
    print(f'   matrix result:   {x_bow_2[0, t].tolist()}')
    print(f'   match: {torch.allclose(x_bow[0, t], x_bow_2[0, t])}')
    print()
```


**Output:**
```
element by element comparison for batch 0

position 0
   for-loop result: [1.9269150495529175, 1.4872841835021973]
   matrix result:   [1.9269150495529175, 1.4872841835021973]
   match: True

position 1
   for-loop result: [1.4138160943984985, -0.3091186285018921]
   matrix result:   [1.4138160943984985, -0.3091186285018921]
   match: True

position 2
   for-loop result: [1.1686835289001465, -0.6175940632820129]
   matrix result:   [1.168683648109436, -0.6175941228866577]
   match: True

position 3
   for-loop result: [0.8657457828521729, -0.8643622994422913]
   matrix result:   [0.8657457828521729, -0.8643622994422913]
   match: True

position 4
   for-loop result: [0.542169451713562, -0.36174526810646057]
   matrix result:   [0.542169451713562, -0.36174529790878296]
   match: True

position 5
   for-loop result: [0.386394739151001, -0.5353888869285583]
   matrix result:   [0.38639479875564575, -0.5353888869285583]
   match: True

position 6
   for-loop result: [0.22721245884895325, -0.5388233065605164]
   matrix result:   [0.22721239924430847, -0.5388233065605164]
   match: True

position 7
   for-loop result: [0.10270603746175766, -0.37616467475891113]
   matrix result:   [0.10270600765943527, -0.3761647045612335]
   match: True


```


### Why Matrix Multiplication is Better for Transformers
| Aspect | For-Loops (Version 1) | Matrix Multiplication (Version 2) |
|--------|----------------------|----------------------------------|
| Speed | Slow (sequential) | Fast (parallel) |
| GPU Friendly | No | Yes |
| Code Length | Long | Short |
| Scalability | Poor | Excellent |
| Memory Access | Random | Contiguous |

Matrix multiplication is the foundation of modern deep learning because it maps perfectly to GPU hardware, which can perform thousands of parallel operations simultaneously.


```python
# final summary: the complete matrix multiplication approach
print('SUMMARY: Matrix Multiplication for Token Averaging')
print('=' * 60)
print()
print('step 1: create lower triangular matrix of ones')
print('        torch.tril(torch.ones(T, T))')
print('        this creates the "only look at past" pattern')
print()
print('step 2: normalize each row to sum to 1')
print('        wei = wei / wei.sum(dim=1, keepdim=True)')
print('        this turns sums into averages')
print()
print('step 3: matrix multiply')
print('        x_bow_2 = wei @ x')
print('        this applies the weighted average in one operation')
print()
print('result: Same as nested for-loops, but much faster!')
print()
print('This is the foundation of self-attention in transformers.')
print('instead of uniform weights (1/n for all), attention learns')
print('which positions to weight more heavily.')
```


**Output:**
```
SUMMARY: Matrix Multiplication for Token Averaging
============================================================

step 1: create lower triangular matrix of ones
        torch.tril(torch.ones(T, T))
        this creates the "only look at past" pattern

step 2: normalize each row to sum to 1
        wei = wei / wei.sum(dim=1, keepdim=True)
        this turns sums into averages

step 3: matrix multiply
        x_bow_2 = wei @ x
        this applies the weighted average in one operation

result: Same as nested for-loops, but much faster!

This is the foundation of self-attention in transformers;
instead of uniform weights (1/n for all), attention learns
which positions to weight more heavily.

```


## MIT License

