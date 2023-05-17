#!/usr/bin/env python3

from scipy.signal import find_peaks
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import joblib
from tqdm.auto import tqdm

# Using joblib, because with lib multiproccessing + tqdm = slowing down by 5 times
from joblib import Parallel, delayed

#h planka
h = 1.0
alpha = 1

# d realization
def d(i, j = None):
    i = int(i)
    if j is None:
        return bin(i).count('1')
    j = int(j)
    res = bin(i^j).count('1')

    return res

def elem(i, j):
    if i == j:
        return 0
    else:
        return 1/d(i, j) * np.exp(complex(0, -alpha*(d(i) - d(j))))

# From A to B sizes
a = 5
b = 6

step = 0.001

# limits to sizes
l = np.array([0, 2, 4, 6, 16, 20, 10, 110, 900, 1500, 800, 2000, 2000])
l = l * np.pi / step
n_proc = joblib.cpu_count()
print(n_proc)

r = np.arange(a, b + 1)
max_probs = np.empty(b + 1 - a)
for size in r:
    n = 2 ** size;

    H = np.empty((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            H[i][j] = elem(i, j)

    evals, evecs = np.linalg.eigh(H)

    evecs_matrix = np.column_stack(evecs).reshape(n, n)
    psi0 = np.zeros(n, dtype=complex)
    psi0[0] = 1

# Start of time
    start = 0
    end = l[size]

# Time list
    tlist = np.arange(0, end) * step

# Coefficient for psi0 in evecs basis
    c = np.empty(n, dtype=complex)
    for i in range(n):
        c[i] = np.dot(evecs_matrix[:, i].conjugate(), psi0)

# Making target state
    target_state = np.zeros(n, dtype=complex)
    target_state[-1] = 1

    probs = np.empty(len(tlist))

# Function of evolution
    def compute_probability(t):
        psi_t = np.dot(evecs_matrix, c * np.exp(-1j * evals * t / h))
        psi_t /= np.linalg.norm(psi_t)
        prob = np.square(np.abs(np.dot(psi_t.conjugate(), target_state)))
        return prob

# Parallel start of compute_probability
    probs = Parallel(n_jobs=n_proc)(
        delayed(compute_probability)(t) for t in tqdm(tlist, desc=f"Size {size}", unit="steps", leave=False)
    )
# Find the maximum probability and corresponding time
    max_prob = np.max(probs)
    print(size, " - ", max_prob)
    max_prob_time = tlist[np.argmax(probs)]
    max_peaks, _ = find_peaks(probs)

# Find the period of the probability oscillation
    periods = np.diff(tlist[max_peaks])
    period = np.mean(periods)

# Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tlist, y=probs, mode='lines', name='Probability'))
    fig.add_trace(go.Scatter(x=[max_prob_time], y=[max_prob], mode='markers', marker=dict(size=10, color='red'), name='Max probability'))

# Set layout of the plot
    fig.update_layout(title='Probability of state ONE: n = ' + str(r),
                      xaxis_title='Time, s * h',
                      yaxis_title='Probability',
                      xaxis_tickvals=[x * np.pi for x in range(int(start), int(end / np.pi) + 1)],
                      xaxis_ticktext=[str(x) + 'π * h' for x in range(int(start), int(end / np.pi) + 1)],
                      showlegend=True,
                      legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0.5)'))

# Add annotations to the plot
    fig.add_annotation(x=max_prob_time, y=max_prob,
                       text='Max probability: {:.3f}'.format(max_prob),
                       showarrow=True,
                       arrowhead=7,
                       ax=0,
                       ay=-50)

# Find the indices of the points that are close to the max_prob
    tolerance = max_prob / 100000
    #tolerance = 0.000001
    close_indices = np.where(np.abs(probs - max_prob) < tolerance)[0]

# Add vertical lines at the x-coordinates of the close points
    for i in close_indices:
        x_val = tlist[i]
        fig.add_shape(dict(type='line', x0=x_val, x1=x_val, y0=0, y1=max_prob, line=dict(color='black', width=1, dash='dot')))

        fig.add_annotation(x=x_val, y=0,
                           text='{}{}'.format(round(x_val / np.pi, 2), 'π * h'),
                           showarrow=False,
                           xanchor='center',
                           yanchor='top',
                           font=dict(size=12, color='black'))

# Save the plot as an interactive HTML file
    pio.write_html(fig, file='interactive_plot_' + str(n) + '.html', auto_open=False)
    max_probs[size - a] = max_prob

# Plot for amplitudes (needs upgrading)
fig = go.Figure()
fig.add_trace(go.Scatter(x=r, y=max_probs, mode='lines+markers+text', text=[f'{val:.3f}' for val in max_probs], textposition='top center'))
fig.update_layout(title='Max probability', xaxis_title='Size', yaxis_title='Max Probability')
pio.write_html(fig, file='max_probs.html', auto_open=False)
