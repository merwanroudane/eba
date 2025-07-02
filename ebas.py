import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64

# Set page configuration
st.set_page_config(
	page_title="Extreme Bounds Analysis (EBA)",
	page_icon="📊",
	layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .concept-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .formula-box {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #FFEDD5;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    .footnote {
        font-size: 0.8rem;
        font-style: italic;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-header'>Understanding Extreme Bounds Analysis (EBA)</div>", unsafe_allow_html=True)
st.markdown("### A Beginner's Guide to Leamer and Sala-i-Martin Methods")

# Introduction
st.markdown("""
This interactive app provides a theoretical understanding of Extreme Bounds Analysis (EBA) 
in econometrics, focusing on the approaches proposed by Edward Leamer and Xavier Sala-i-Martin. 
EBA is a sensitivity analysis technique that helps determine the robustness of variables in regression models.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
	"Go to",
	["Introduction to EBA",
	 "Leamer's EBA Method",
	 "Sala-i-Martin's Method",
	 "Comparing the Methods",
	 "Practical Applications",
	 "Limitations and Critiques"]
)

# Introduction to EBA
if section == "Introduction to EBA":
	st.markdown("<div class='sub-header'>Introduction to Extreme Bounds Analysis</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>What is Extreme Bounds Analysis?</b><br>
    Extreme Bounds Analysis (EBA) is a sensitivity analysis method used to identify "robust" determinants 
    in empirical models, particularly in economics and cross-country growth regressions. It helps address 
    the problem of model uncertainty, where researchers are uncertain about which variables should be 
    included in their regression models.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("#### The Problem of Model Uncertainty")

	st.markdown("""
    In empirical research, particularly in economics, researchers often face a fundamental problem:

    * A large set of potential explanatory variables
    * Limited theoretical guidance on which variables to include
    * The risk of omitted variable bias
    * The danger of "fishing" for desired results by selective inclusion of variables

    This leads to what economists call the "model uncertainty problem" - when we're unsure which variables 
    should be in our model, results can vary dramatically based on model specification.
    """)

	col1, col2 = st.columns(2)

	with col1:
		# Create figure showing model uncertainty
		variables = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
		models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

		# Create random model specifications (which variables are included)
		np.random.seed(42)
		model_matrix = np.random.choice([0, 1], size=(len(models), len(variables)), p=[0.6, 0.4])

		# Convert to DataFrame for visualization
		model_df = pd.DataFrame(model_matrix, columns=variables, index=models)

		# Create heatmap
		fig = px.imshow(model_df,
						labels=dict(x="Variables", y="Model Specifications", color="Included"),
						x=variables,
						y=models,
						color_continuous_scale=["white", "#3B82F6"],
						title="Different Model Specifications")

		fig.update_layout(height=400)
		st.plotly_chart(fig, use_container_width=True)

	with col2:
		# Create figure showing varying coefficient estimates
		# Sample coefficients for a variable across different model specifications
		np.random.seed(123)
		coefficients = np.random.normal(0.5, 0.3, 20)
		std_errors = np.random.uniform(0.1, 0.2, 20)
		lower_ci = coefficients - 1.96 * std_errors
		upper_ci = coefficients + 1.96 * std_errors

		fig = go.Figure()

		# Add confidence intervals
		for i in range(20):
			fig.add_trace(go.Scatter(
				x=[i, i],
				y=[lower_ci[i], upper_ci[i]],
				mode='lines',
				line=dict(color='rgba(59, 130, 246, 0.5)', width=2),
				showlegend=False
			))

		# Add coefficients
		fig.add_trace(go.Scatter(
			x=list(range(20)),
			y=coefficients,
			mode='markers',
			marker=dict(color='#3B82F6', size=8),
			name='Coefficient'
		))

		# Add zero line
		fig.add_shape(
			type='line',
			x0=-1, y0=0,
			x1=20, y1=0,
			line=dict(color='red', dash='dash')
		)

		fig.update_layout(
			title="Variable Coefficient Across Different Model Specifications",
			xaxis_title="Model Specification",
			yaxis_title="Coefficient Value",
			height=400
		)

		st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figures above illustrate the problem of model uncertainty. The left figure shows how researchers 
    might include different sets of variables in their models. The right figure shows how the coefficient 
    for a single variable can vary dramatically across different model specifications, sometimes even 
    changing sign.

    This variability raises a critical question: **Which results should we trust?**
    """)

	st.markdown("#### The Goal of Extreme Bounds Analysis")

	st.markdown("""
    <div class="concept-box">
    EBA aims to determine whether the relationship between a dependent variable (Y) and an independent 
    variable of interest (X) is <span class="highlight">robust</span> to changes in the set of control 
    variables included in the regression.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    EBA involves running many regressions with different combinations of control variables and examining 
    how the coefficient of the variable of interest changes. If the coefficient remains significant and 
    of the same sign across all (or most) specifications, it is considered "robust."
    """)

	st.markdown("<div class='section-header'>The Basic Framework</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="formula-box">
    The general model in EBA is:
    """, unsafe_allow_html=True)

	st.latex(r'''
    Y = \alpha + \beta_M M + \beta_F F + \beta_Z Z + \varepsilon
    ''')

	st.markdown("""
    Where:
    * $Y$ is the dependent variable
    * $M$ is the variable of interest (whose robustness we want to test)
    * $F$ is a set of "free" variables that are always included in the regression
    * $Z$ is a subset of variables chosen from a larger set of potential control variables
    * $\varepsilon$ is the error term
    </div>
    """, unsafe_allow_html=True)

	# Visualization of the general EBA approach
	st.markdown("#### The EBA Process")

	col1, col2 = st.columns([2, 1])

	with col1:
		# Create a flowchart-like visualization
		nodes = ['Define Variables of Interest (M)',
				 'Identify Fixed Variables (F)',
				 'Specify Pool of Control Variables (Z)',
				 'Run Many Regressions with Different Z Combinations',
				 'Examine Distribution of Coefficients',
				 'Apply Robustness Criteria',
				 'Identify Robust Variables']

		y_pos = np.arange(len(nodes))
		x_pos = [0] * len(nodes)

		colors = ['#DBEAFE'] * len(nodes)
		colors[0] = '#93C5FD'  # Start node
		colors[-1] = '#3B82F6'  # End node

		fig = go.Figure(data=[go.Bar(
			x=[1] * len(nodes),
			y=nodes,
			orientation='h',
			marker=dict(color=colors),
			width=0.6
		)])

		# Add arrows
		for i in range(len(nodes) - 1):
			fig.add_shape(
				type="line",
				x0=0.5, y0=len(nodes) - i - 1.3,
				x1=0.5, y1=len(nodes) - i - 1.7,
				line=dict(color="#1E3A8A", width=2),
				xref="x", yref="y"
			)

			# Add arrowhead
			fig.add_shape(
				type="path",
				path=" M 0.4,{0} L 0.5,{1} L 0.6,{0} Z".format(len(nodes) - i - 1.7, len(nodes) - i - 1.8),
				fillcolor="#1E3A8A",
				line=dict(color="#1E3A8A"),
				xref="x", yref="y"
			)

		fig.update_layout(
			title="EBA Process Flowchart",
			xaxis=dict(
				showticklabels=False,
				showgrid=False,
				zeroline=False,
				range=[0, 1]
			),
			yaxis=dict(
				showticklabels=True,
				showgrid=False,
				zeroline=False,
				autorange="reversed"
			),
			margin=dict(l=0, r=0, t=50, b=0),
			height=500,
			plot_bgcolor='rgba(0,0,0,0)'
		)

		st.plotly_chart(fig, use_container_width=True)

	with col2:
		st.markdown("""
        ### Key Components of EBA

        **1. Focus Variable (M)**
        - The variable whose robustness we want to test

        **2. Free Variables (F)**
        - Always included in all regressions
        - Usually strongly justified by theory

        **3. Doubtful Variables (Z)**
        - Pool of control variables
        - Different combinations tested

        **4. Model Combinations**
        - Systematic testing of different model specifications

        **5. Robustness Criteria**
        - Rules to determine which variables are "robust"
        - Varies between Leamer and Sala-i-Martin approaches
        """)

# Leamer's EBA Method
elif section == "Leamer's EBA Method":
	st.markdown("<div class='sub-header'>Leamer's Extreme Bounds Analysis</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>Leamer's EBA Approach (1983, 1985)</b><br>
    Edward Leamer proposed the original Extreme Bounds Analysis as a way to address model uncertainty 
    in econometrics. His approach focuses on identifying the extreme bounds (maximum and minimum values) 
    of coefficient estimates across all possible model specifications.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("#### Theoretical Foundation")

	st.markdown("""
    Leamer's approach stems from a Bayesian perspective on model uncertainty. He argued that researchers 
    often implicitly use prior information when selecting variables for their models, which can lead to 
    specification bias.

    The key insight of Leamer's approach is that if a variable's effect changes drastically when we change 
    model specification, we should be skeptical about its true relationship with the dependent variable.
    """)

	st.markdown("<div class='section-header'>Methodology</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="formula-box">
    Leamer's EBA involves running regressions of the form:
    """, unsafe_allow_html=True)

	st.latex(r'''
    Y = \alpha + \beta_M M + \beta_F F + \beta_Z Z + \varepsilon
    ''')

	st.markdown("""
    For each variable M whose robustness we want to test, we run many regressions with different 
    combinations of control variables (Z) from the set of doubtful variables.
    </div>
    """, unsafe_allow_html=True)

	col1, col2 = st.columns([3, 2])

	with col1:
		# Create a visual showing how extreme bounds are calculated
		np.random.seed(42)

		# Simulate coefficient estimates from different model specifications
		beta_estimates = np.random.normal(0.5, 0.2, 50)
		std_errors = np.random.uniform(0.05, 0.15, 50)
		lower_bounds = beta_estimates - 2 * std_errors
		upper_bounds = beta_estimates + 2 * std_errors

		# Find the extreme bounds
		extreme_lower = min(lower_bounds)
		extreme_upper = max(upper_bounds)

		# Create the figure
		fig = go.Figure()

		# Add confidence intervals for each model
		for i in range(50):
			fig.add_trace(go.Scatter(
				x=[i, i],
				y=[lower_bounds[i], upper_bounds[i]],
				mode='lines',
				line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
				showlegend=False
			))

		# Add coefficient points
		fig.add_trace(go.Scatter(
			x=list(range(50)),
			y=beta_estimates,
			mode='markers',
			marker=dict(color='#3B82F6', size=6),
			name='β Estimates'
		))

		# Add extreme bounds
		fig.add_shape(
			type='line',
			x0=-1, y0=extreme_lower,
			x1=50, y1=extreme_lower,
			line=dict(color='red', width=2, dash='dash')
		)

		fig.add_shape(
			type='line',
			x0=-1, y0=extreme_upper,
			x1=50, y1=extreme_upper,
			line=dict(color='red', width=2, dash='dash')
		)

		# Add zero line
		fig.add_shape(
			type='line',
			x0=-1, y0=0,
			x1=50, y1=0,
			line=dict(color='black', dash='dot')
		)

		# Add annotations for extreme bounds
		fig.add_annotation(
			x=49, y=extreme_lower,
			text="Extreme Lower Bound",
			showarrow=True,
			arrowhead=1,
			ax=40,
			ay=20
		)

		fig.add_annotation(
			x=49, y=extreme_upper,
			text="Extreme Upper Bound",
			showarrow=True,
			arrowhead=1,
			ax=40,
			ay=-20
		)

		fig.update_layout(
			title="Leamer's EBA: Finding Extreme Bounds",
			xaxis_title="Different Model Specifications",
			yaxis_title="Coefficient of Variable M (β)",
			height=400,
			xaxis=dict(showticklabels=False)
		)

		st.plotly_chart(fig, use_container_width=True)

	with col2:
		st.markdown("""
        ### Steps in Leamer's EBA

        **1. Define Variables**
        - M: Variable of interest
        - F: Fixed/free variables (always included)
        - Z: Pool of doubtful variables

        **2. Run Regressions**
        - Test all (or many) possible combinations of Z variables
        - For each regression, record coefficient and standard error for M

        **3. Calculate Extreme Bounds**
        - Lower extreme bound: 
          Lowest β - 2×(standard error)
        - Upper extreme bound: 
          Highest β + 2×(standard error)

        **4. Apply Robustness Criterion**
        - If extreme bounds have the same sign (both positive or both negative)
        - AND both bounds are statistically significant
        - THEN variable M is considered "robust"
        """)

	st.markdown("<div class='section-header'>Robustness Criteria in Leamer's EBA</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="formula-box">
    For a variable to be considered "robust" under Leamer's criteria:
    """, unsafe_allow_html=True)

	st.latex(r'''
    \text{sign}(\beta_{min} - 2\sigma_{min}) = \text{sign}(\beta_{max} + 2\sigma_{max}) \neq 0
    ''')

	st.markdown("""
    Where:
    * $\beta_{min}$ is the smallest estimated coefficient for M across all models
    * $\beta_{max}$ is the largest estimated coefficient for M across all models
    * $\sigma_{min}$ and $\sigma_{max}$ are their respective standard errors
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    In other words, the extreme bounds (typically calculated as coefficient ± 2 standard errors) 
    must have the same sign and be statistically significantly different from zero.
    """)

	# Create a visual demonstration of robust vs. non-robust variables
	st.markdown("#### Examples of Robust vs. Non-Robust Variables")

	col1, col2 = st.columns(2)

	with col1:
		# Example of a robust variable
		np.random.seed(123)

		# Generate coefficients that are all positive
		beta_robust = np.random.normal(0.6, 0.1, 30)
		se_robust = np.random.uniform(0.1, 0.15, 30)
		lower_robust = beta_robust - 2 * se_robust
		upper_robust = beta_robust + 2 * se_robust

		# Create figure
		fig = go.Figure()

		# Add confidence intervals
		for i in range(30):
			fig.add_trace(go.Scatter(
				x=[i, i],
				y=[lower_robust[i], upper_robust[i]],
				mode='lines',
				line=dict(color='rgba(16, 185, 129, 0.3)', width=1),
				showlegend=False
			))

		# Add coefficient points
		fig.add_trace(go.Scatter(
			x=list(range(30)),
			y=beta_robust,
			mode='markers',
			marker=dict(color='#10B981', size=6),
			name='β Estimates'
		))

		# Add extreme bounds
		min_lower = min(lower_robust)
		max_upper = max(upper_robust)

		fig.add_shape(
			type='line',
			x0=-1, y0=min_lower,
			x1=30, y1=min_lower,
			line=dict(color='#10B981', width=2, dash='dash')
		)

		fig.add_shape(
			type='line',
			x0=-1, y0=max_upper,
			x1=30, y1=max_upper,
			line=dict(color='#10B981', width=2, dash='dash')
		)

		# Add zero line
		fig.add_shape(
			type='line',
			x0=-1, y0=0,
			x1=30, y1=0,
			line=dict(color='black', dash='dot')
		)

		fig.update_layout(
			title="Robust Variable Example",
			xaxis_title="Model Specifications",
			yaxis_title="Coefficient (β)",
			height=350,
			xaxis=dict(showticklabels=False)
		)

		# Add annotation
		fig.add_annotation(
			x=15, y=max_upper + 0.1,
			text="ROBUST: Both extreme bounds are positive",
			showarrow=False,
			font=dict(color="#10B981", size=12)
		)

		st.plotly_chart(fig, use_container_width=True)

	with col2:
		# Example of a non-robust variable
		np.random.seed(456)

		# Generate coefficients that cross zero
		beta_nonrobust = np.random.normal(0.3, 0.3, 30)
		se_nonrobust = np.random.uniform(0.1, 0.2, 30)
		lower_nonrobust = beta_nonrobust - 2 * se_nonrobust
		upper_nonrobust = beta_nonrobust + 2 * se_nonrobust

		# Create figure
		fig = go.Figure()

		# Add confidence intervals
		for i in range(30):
			fig.add_trace(go.Scatter(
				x=[i, i],
				y=[lower_nonrobust[i], upper_nonrobust[i]],
				mode='lines',
				line=dict(color='rgba(239, 68, 68, 0.3)', width=1),
				showlegend=False
			))

		# Add coefficient points
		fig.add_trace(go.Scatter(
			x=list(range(30)),
			y=beta_nonrobust,
			mode='markers',
			marker=dict(color='#EF4444', size=6),
			name='β Estimates'
		))

		# Add extreme bounds
		min_lower = min(lower_nonrobust)
		max_upper = max(upper_nonrobust)

		fig.add_shape(
			type='line',
			x0=-1, y0=min_lower,
			x1=30, y1=min_lower,
			line=dict(color='#EF4444', width=2, dash='dash')
		)

		fig.add_shape(
			type='line',
			x0=-1, y0=max_upper,
			x1=30, y1=max_upper,
			line=dict(color='#EF4444', width=2, dash='dash')
		)

		# Add zero line
		fig.add_shape(
			type='line',
			x0=-1, y0=0,
			x1=30, y1=0,
			line=dict(color='black', dash='dot')
		)

		fig.update_layout(
			title="Non-Robust Variable Example",
			xaxis_title="Model Specifications",
			yaxis_title="Coefficient (β)",
			height=350,
			xaxis=dict(showticklabels=False)
		)

		# Add annotation
		fig.add_annotation(
			x=15, y=min_lower - 0.1,
			text="NOT ROBUST: Extreme bounds have different signs",
			showarrow=False,
			font=dict(color="#EF4444", size=12)
		)

		st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='section-header'>Limitations of Leamer's Approach</div>", unsafe_allow_html=True)

	st.markdown("""
    While groundbreaking, Leamer's EBA has some significant limitations:

    1. **Extremely Stringent Criteria**: Very few variables pass the robustness test, as the extreme bounds 
       often cross zero even for important variables.

    2. **All Model Specifications Are Treated Equally**: The method gives equal weight to all model 
       specifications, even those that might be theoretically implausible.

    3. **Computational Intensity**: With k potential control variables, there are 2^k possible model 
       combinations, which can be computationally demanding.

    4. **Outlier Sensitivity**: A single extreme model specification can determine the extreme bounds, 
       making the approach sensitive to outliers.

    These limitations led to the development of modified approaches, most notably Sala-i-Martin's EBA method.
    """)

# Sala-i-Martin's Method
elif section == "Sala-i-Martin's Method":
	st.markdown("<div class='sub-header'>Sala-i-Martin's Approach to EBA</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>Sala-i-Martin's Approach (1997)</b><br>
    Xavier Sala-i-Martin proposed a less extreme version of EBA in his 1997 paper "I Just Ran Two Million Regressions." 
    Instead of focusing on extreme bounds, his approach examines the entire distribution of coefficient estimates 
    across model specifications.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("#### Motivation")

	st.markdown("""
    Sala-i-Martin argued that Leamer's approach was too stringent - very few variables could pass the 
    extreme bounds test, even variables that seemed economically important. He proposed a less restrictive 
    approach that considered the entire distribution of coefficient estimates rather than just the extremes.

    His key insight was that we should care about the proportion of the distribution of coefficient estimates 
    that lies on each side of zero, rather than just the extreme bounds.
    """)

	st.markdown("<div class='section-header'>Methodology</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="formula-box">
    Like Leamer's approach, Sala-i-Martin's method involves running many regressions of the form:
    """, unsafe_allow_html=True)

	st.latex(r'''
    Y = \alpha + \beta_M M + \beta_F F + \beta_Z Z + \varepsilon
    ''')

	st.markdown("""
    But instead of just looking at extreme bounds, Sala-i-Martin examines the entire distribution of β coefficients.
    </div>
    """, unsafe_allow_html=True)

	col1, col2 = st.columns([3, 2])

	with col1:
		# Create a visual showing the distribution of coefficients
		np.random.seed(42)

		# Generate two sets of coefficients - one mostly positive, one mixed
		robust_coefs = np.random.normal(0.5, 0.15, 500)
		nonrobust_coefs = np.random.normal(0.1, 0.3, 500)

		# Create the figure with subplots
		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Robust Variable: 95% of distribution > 0",
											"Non-Robust Variable: Only 65% of distribution > 0"))

		# Add histograms
		fig.add_trace(
			go.Histogram(
				x=robust_coefs,
				nbinsx=30,
				marker_color='rgba(16, 185, 129, 0.7)',
				name="Robust Variable"
			),
			row=1, col=1
		)

		fig.add_trace(
			go.Histogram(
				x=nonrobust_coefs,
				nbinsx=30,
				marker_color='rgba(239, 68, 68, 0.7)',
				name="Non-Robust Variable"
			),
			row=2, col=1
		)

		# Add vertical line at zero
		fig.add_shape(
			type='line',
			x0=0, y0=0,
			x1=0, y1=60,
			line=dict(color='black', dash='dash'),
			row=1, col=1
		)

		fig.add_shape(
			type='line',
			x0=0, y0=0,
			x1=0, y1=60,
			line=dict(color='black', dash='dash'),
			row=2, col=1
		)

		# Calculate and display CDF at zero
		pos_pct_robust = (robust_coefs > 0).mean() * 100
		pos_pct_nonrobust = (nonrobust_coefs > 0).mean() * 100

		fig.add_annotation(
			x=0.5, y=50,
			text=f"{pos_pct_robust:.1f}% of coefficients > 0",
			showarrow=False,
			font=dict(color="#10B981"),
			row=1, col=1
		)

		fig.add_annotation(
			x=0.5, y=50,
			text=f"{pos_pct_nonrobust:.1f}% of coefficients > 0",
			showarrow=False,
			font=dict(color="#EF4444"),
			row=2, col=1
		)

		fig.update_layout(
			title="Sala-i-Martin's Approach: Distribution of Coefficients",
			height=500,
			showlegend=False
		)

		st.plotly_chart(fig, use_container_width=True)

	with col2:
		st.markdown("""
        ### Steps in Sala-i-Martin's EBA

        **1. Define Variables**
        - Same as Leamer's approach: M, F, and Z variables

        **2. Run Many Regressions**
        - Test many different combinations of Z variables
        - For each regression, record coefficient for M

        **3. Analyze Distribution**
        - Look at the entire distribution of coefficient estimates
        - Calculate what fraction of the distribution is greater than zero (for positive relationships) or less than zero (for negative relationships)

        **4. Apply Robustness Criterion**
        - If a large fraction (e.g., 95%) of the distribution lies on one side of zero
        - THEN variable M is considered "robust"

        **5. Weighted vs. Unweighted**
        - Sala-i-Martin proposed both weighted and unweighted approaches
        - Weighted: each coefficient is weighted by the likelihood of the model
        - Unweighted: all models receive equal weight
        """)

	st.markdown("<div class='section-header'>Robustness Criteria in Sala-i-Martin's Approach</div>",
				unsafe_allow_html=True)

	st.markdown("""
    <div class="formula-box">
    Sala-i-Martin proposed two main approaches to determine robustness:
    """, unsafe_allow_html=True)

	st.latex(r'''
    \text{CDF}(0) = \int_{-\infty}^{0} f(\beta)d\beta \quad \text{or} \quad \text{CDF}(0) = \int_{0}^{\infty} f(\beta)d\beta
    ''')

	st.markdown("""
    Where $f(\beta)$ is the distribution of coefficient estimates across models.

    A variable is considered robust if:
    * For expected positive relationships: $\text{CDF}(0) \leq 0.05$ (95% of distribution is above zero)
    * For expected negative relationships: $\text{CDF}(0) \geq 0.95$ (95% of distribution is below zero)
    </div>
    """, unsafe_allow_html=True)

	st.markdown("#### Normal vs. Non-Normal Distributions")

	col1, col2 = st.columns(2)

	with col1:
		# Normal distribution example
		x = np.linspace(-1, 1.5, 1000)
		y = np.exp(-(x - 0.5) ** 2 / 0.05) / np.sqrt(2 * np.pi * 0.05)

		fig = go.Figure()

		# Add the PDF curve
		fig.add_trace(go.Scatter(
			x=x, y=y,
			mode='lines',
			line=dict(color='#3B82F6', width=2),
			name='PDF',
			fill='tozeroy',
			fillcolor='rgba(59, 130, 246, 0.2)'
		))

		# Add vertical line at zero
		fig.add_shape(
			type='line',
			x0=0, y0=0,
			x1=0, y1=2,
			line=dict(color='red', dash='dash')
		)

		# Shade area to the left of zero
		x_left = x[x < 0]
		y_left = y[x < 0]

		fig.add_trace(go.Scatter(
			x=x_left, y=y_left,
			mode='lines',
			line=dict(width=0),
			fill='tozeroy',
			fillcolor='rgba(239, 68, 68, 0.5)',
			name='CDF(0) = 0.02'
		))

		fig.update_layout(
			title="Normal Distribution Approach",
			xaxis_title="Coefficient Value (β)",
			yaxis_title="Density",
			height=350
		)

		st.plotly_chart(fig, use_container_width=True)

	with col2:
		# Non-normal distribution example
		# Create a bimodal distribution
		x = np.linspace(-1, 1.5, 1000)
		y1 = np.exp(-(x + 0.2) ** 2 / 0.05) / np.sqrt(2 * np.pi * 0.05) * 0.3
		y2 = np.exp(-(x - 0.6) ** 2 / 0.07) / np.sqrt(2 * np.pi * 0.07) * 0.7
		y = y1 + y2

		fig = go.Figure()

		# Add the PDF curve
		fig.add_trace(go.Scatter(
			x=x, y=y,
			mode='lines',
			line=dict(color='#8B5CF6', width=2),
			name='PDF',
			fill='tozeroy',
			fillcolor='rgba(139, 92, 246, 0.2)'
		))

		# Add vertical line at zero
		fig.add_shape(
			type='line',
			x0=0, y0=0,
			x1=0, y1=2,
			line=dict(color='red', dash='dash')
		)

		# Shade area to the left of zero
		x_left = x[x < 0]
		y_left = y[x < 0]

		fig.add_trace(go.Scatter(
			x=x_left, y=y_left,
			mode='lines',
			line=dict(width=0),
			fill='tozeroy',
			fillcolor='rgba(239, 68, 68, 0.5)',
			name='CDF(0) = 0.25'
		))

		fig.update_layout(
			title="Non-Normal Distribution Case",
			xaxis_title="Coefficient Value (β)",
			yaxis_title="Density",
			height=350
		)

		st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    Sala-i-Martin proposed two approaches to analyzing the distribution:

    1. **Normal Distribution Approach**: Assume the distribution of coefficients across models follows a normal 
    distribution. Calculate the mean and standard deviation, then determine the CDF at zero.

    2. **Non-Parametric Approach**: Make no assumptions about the distribution shape. Simply count what 
    fraction of the estimated coefficients are positive and what fraction are negative.
    """)

	st.markdown("<div class='section-header'>Weighting Model Specifications</div>", unsafe_allow_html=True)

	st.markdown("""
    Another key innovation in Sala-i-Martin's approach is weighting different model specifications by their likelihood:

    1. **Unweighted Approach**: All model specifications receive equal weight

    2. **Weighted Approach**: Models are weighted by their likelihood (how well they fit the data)
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="formula-box">
    In the weighted approach, the overall CDF is calculated as:
    """, unsafe_allow_html=True)

	st.latex(r'''
    \text{CDF}(0) = \sum_{j=1}^{M} \omega_j \cdot \text{CDF}_j(0)
    ''')

	st.markdown("""
    Where:
    * $\text{CDF}_j(0)$ is the CDF at zero for model $j$
    * $\omega_j$ is the weight of model $j$ (typically related to the model's goodness of fit)
    * $M$ is the total number of models
    </div>
    """, unsafe_allow_html=True)

	# Create a visual example of weighted vs unweighted approach
	np.random.seed(123)

	# Generate coefficient estimates for 10 models
	coefs = np.array([0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, 0.2, 0.1, 0.05])
	std_errors = np.array([0.1, 0.15, 0.1, 0.2, 0.15, 0.05, 0.1, 0.1, 0.05, 0.05])

	# Generate R-squared values for weighting
	r_squared = np.array([0.8, 0.75, 0.85, 0.7, 0.9, 0.4, 0.3, 0.5, 0.45, 0.6])
	weights = r_squared / r_squared.sum()

	# Equal weights
	equal_weights = np.ones_like(coefs) / len(coefs)

	# Create comparison figure
	fig = go.Figure()

	# Add bar for each model
	models = [f"Model {i + 1}" for i in range(len(coefs))]

	# Add bars for coefficients
	fig.add_trace(go.Bar(
		x=models,
		y=coefs,
		name='Coefficient',
		marker_color=np.where(coefs > 0, 'rgba(16, 185, 129, 0.7)', 'rgba(239, 68, 68, 0.7)'),
		error_y=dict(
			type='data',
			array=std_errors * 1.96,
			visible=True
		)
	))

	# Add markers for weights
	fig.add_trace(go.Scatter(
		x=models,
		y=weights * 3,  # Scale up for visibility
		mode='markers',
		marker=dict(
			symbol='diamond',
			size=12,
			color='rgba(59, 130, 246, 0.8)',
			line=dict(color='rgb(8, 48, 107)', width=1)
		),
		name='Model Weight'
	))

	# Add horizontal line at zero
	fig.add_shape(
		type='line',
		x0=-0.5, y0=0,
		x1=9.5, y1=0,
		line=dict(color='black', dash='dash')
	)

	# Add annotations
	unweighted_positive = (coefs > 0).mean() * 100
	weighted_positive = np.sum(weights[coefs > 0]) * 100

	fig.add_annotation(
		x=4.5, y=0.85,
		text=f"Unweighted: {unweighted_positive:.1f}% positive coefficients",
		showarrow=False,
		bgcolor="rgba(255, 255, 255, 0.8)"
	)

	fig.add_annotation(
		x=4.5, y=0.75,
		text=f"Weighted: {weighted_positive:.1f}% weight on positive coefficients",
		showarrow=False,
		bgcolor="rgba(255, 255, 255, 0.8)"
	)

	fig.update_layout(
		title="Weighted vs. Unweighted Approach",
		xaxis_title="Model Specification",
		yaxis_title="Coefficient Value (β)",
		height=450,
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates how weighting can affect the robustness conclusion. In this example:

    - Using the unweighted approach, 70% of coefficients are positive.
    - Using the weighted approach (by R-squared), about 90% of the weighted distribution is positive.

    This shows how weighting can change our conclusions about robustness, especially when better-fitting 
    models tend to produce similar coefficient estimates.
    """)

	st.markdown("<div class='section-header'>Advantages of Sala-i-Martin's Approach</div>", unsafe_allow_html=True)

	st.markdown("""
    Compared to Leamer's original EBA, Sala-i-Martin's approach offers several advantages:

    1. **Less Stringent Criteria**: By looking at the entire distribution rather than extreme bounds, 
       more variables can be identified as robust.

    2. **Accounts for Model Quality**: The weighted approach gives more importance to models that fit the 
       data better.

    3. **Provides More Information**: Instead of a binary robust/not-robust classification, it quantifies 
       the degree of robustness (e.g., "95% of the distribution is positive").

    4. **Less Sensitive to Outliers**: A single extreme model specification has less influence on the 
       overall conclusion.

    5. **Handles Non-Normal Distributions**: The non-parametric approach makes no assumptions about the 
       shape of the coefficient distribution.
    """)

# Comparing the Methods
elif section == "Comparing the Methods":
	st.markdown("<div class='sub-header'>Comparing Leamer's and Sala-i-Martin's Approaches</div>",
				unsafe_allow_html=True)

	st.markdown("""
    Both Leamer's and Sala-i-Martin's approaches to Extreme Bounds Analysis aim to address model uncertainty, 
    but they differ in their methodologies and criteria for robustness. This section compares the two approaches 
    and discusses their relative strengths and weaknesses.
    """)

	# Create a comparison table
	comparison_data = {
		"Feature": [
			"Focus of Analysis",
			"Robustness Criteria",
			"Treatment of Model Specifications",
			"Sensitivity to Outliers",
			"Computational Intensity",
			"Typical Results",
			"Information Provided",
			"Theoretical Foundation"
		],
		"Leamer's EBA": [
			"Extreme bounds of coefficient estimates",
			"Extreme bounds must have same sign and be significant",
			"All specifications treated equally",
			"Very sensitive - one extreme model determines bounds",
			"Lower - only need to find min/max bounds",
			"Few variables pass the robustness test",
			"Binary classification (robust/not robust)",
			"Bayesian approach to specification uncertainty"
		],
		"Sala-i-Martin's Approach": [
			"Entire distribution of coefficient estimates",
			"Large portion (e.g., 95%) of distribution on one side of zero",
			"Can weight specifications by likelihood/goodness of fit",
			"Less sensitive - based on entire distribution",
			"Higher - need to analyze full distribution",
			"More variables identified as robust",
			"Quantitative measure of robustness (% of distribution)",
			"Modified classical approach with Bayesian elements"
		]
	}

	comparison_df = pd.DataFrame(comparison_data)


	# Style the table
	def highlight_row(s):
		return ['background-color: #EFF6FF' if i % 2 == 0 else '' for i in range(len(s))]


	styled_df = comparison_df.style.apply(highlight_row, axis=1)

	st.markdown("<div class='section-header'>Key Differences</div>", unsafe_allow_html=True)
	st.table(styled_df)

	st.markdown("<div class='section-header'>Visual Comparison of Methods</div>", unsafe_allow_html=True)

	# Generate sample data for visualization
	np.random.seed(42)

	# Generate coefficient estimates from different model specifications
	coefs = np.random.normal(0.3, 0.2, 100)
	std_errors = np.random.uniform(0.05, 0.15, 100)
	lower_bounds = coefs - 2 * std_errors
	upper_bounds = coefs + 2 * std_errors

	# Calculate extreme bounds
	extreme_lower = min(lower_bounds)
	extreme_upper = max(upper_bounds)

	# Create the figure
	fig = make_subplots(
		rows=2, cols=1,
		subplot_titles=("Leamer's Approach: Focus on Extreme Bounds",
						"Sala-i-Martin's Approach: Distribution Analysis"),
		vertical_spacing=0.15
	)

	# Top plot: Leamer's approach
	# Add confidence intervals for each model
	for i in range(20):  # Just show 20 for clarity
		fig.add_trace(
			go.Scatter(
				x=[i, i],
				y=[lower_bounds[i], upper_bounds[i]],
				mode='lines',
				line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
				showlegend=False
			),
			row=1, col=1
		)

	# Add coefficient points
	fig.add_trace(
		go.Scatter(
			x=list(range(20)),
			y=coefs[:20],
			mode='markers',
			marker=dict(color='#3B82F6', size=6),
			name='β Estimates',
			showlegend=False
		),
		row=1, col=1
	)

	# Add extreme bounds
	fig.add_shape(
		type='line',
		x0=-1, y0=extreme_lower,
		x1=20, y1=extreme_lower,
		line=dict(color='red', width=2, dash='dash'),
		row=1, col=1
	)

	fig.add_shape(
		type='line',
		x0=-1, y0=extreme_upper,
		x1=20, y1=extreme_upper,
		line=dict(color='red', width=2, dash='dash'),
		row=1, col=1
	)

	# Add zero line
	fig.add_shape(
		type='line',
		x0=-1, y0=0,
		x1=20, y1=0,
		line=dict(color='black', dash='dot'),
		row=1, col=1
	)

	# Add annotations for extreme bounds
	fig.add_annotation(
		x=19, y=extreme_lower,
		text="Extreme Lower Bound",
		showarrow=True,
		arrowhead=1,
		ax=40,
		ay=20,
		row=1, col=1
	)

	fig.add_annotation(
		x=19, y=extreme_upper,
		text="Extreme Upper Bound",
		showarrow=True,
		arrowhead=1,
		ax=40,
		ay=-20,
		row=1, col=1
	)

	# Bottom plot: Sala-i-Martin's approach
	# Add histogram of coefficients
	fig.add_trace(
		go.Histogram(
			x=coefs,
			nbinsx=30,
			marker_color='rgba(16, 185, 129, 0.7)',
			name="Coefficient Distribution",
			showlegend=False
		),
		row=2, col=1
	)

	# Add vertical line at zero
	fig.add_shape(
		type='line',
		x0=0, y0=0,
		x1=0, y1=25,
		line=dict(color='red', dash='dash'),
		row=2, col=1
	)

	# Calculate percentage positive and negative
	pos_pct = (coefs > 0).mean() * 100

	# Add annotation
	fig.add_annotation(
		x=0.6, y=20,
		text=f"{pos_pct:.1f}% of distribution > 0",
		showarrow=False,
		font=dict(color="#10B981"),
		row=2, col=1
	)

	fig.add_annotation(
		x=-0.2, y=5,
		text=f"{100 - pos_pct:.1f}% of distribution < 0",
		showarrow=False,
		font=dict(color="#EF4444"),
		row=2, col=1
	)

	# Update layout
	fig.update_layout(
		height=700,
		xaxis=dict(title="Model Specifications", showticklabels=False),
		xaxis2=dict(title="Coefficient Value (β)"),
		yaxis=dict(title="Coefficient Value (β)"),
		yaxis2=dict(title="Frequency")
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates the fundamental difference between the two approaches:

    - **Leamer's Approach (Top)**: Focuses on the extreme bounds - the lowest lower bound and highest upper bound 
      across all model specifications. In this example, the extreme bounds cross zero, so the variable would be 
      classified as not robust.

    - **Sala-i-Martin's Approach (Bottom)**: Examines the entire distribution of coefficient estimates. 
      In this example, about 91% of the distribution is positive, so the variable might be classified as robust 
      if using a 90% threshold.
    """)

	st.markdown("<div class='section-header'>When to Use Each Approach</div>", unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
        ### Consider Leamer's Approach When:

        - You need a very stringent test of robustness
        - You want to be extremely conservative in your claims
        - You are primarily concerned with establishing bounds rather than point estimates
        - The research question demands high certainty
        - You have strong theoretical priors about which variables should be included
        """)

	with col2:
		st.markdown("""
        ### Consider Sala-i-Martin's Approach When:

        - Leamer's criteria is too stringent for your research question
        - You want to quantify the degree of robustness
        - You believe some model specifications are more plausible than others
        - You're interested in the overall pattern rather than extreme cases
        - You're conducting exploratory research
        """)

	st.markdown("<div class='section-header'>Practical Example: Growth Regressions</div>", unsafe_allow_html=True)

	st.markdown("""
    A famous application of these methods is in cross-country growth regressions, where researchers try to 
    identify factors that consistently explain differences in economic growth rates across countries.

    Sala-i-Martin's 1997 paper "I Just Ran Two Million Regressions" applied his approach to growth determinants 
    and found:
    """)

	# Create a figure showing results from Sala-i-Martin's paper
	variables = [
		"Initial Income Level",
		"Life Expectancy",
		"Primary School Enrollment",
		"Investment Rate",
		"Fraction Confucian",
		"Rule of Law",
		"Fraction Muslim",
		"Political Rights",
		"Latin America Dummy",
		"Sub-Saharan Africa Dummy"
	]

	# Made up data for illustration
	robust_pct = [1.000, 0.994, 0.992, 0.982, 0.975, 0.964, 0.948, 0.853, 0.825, 0.802]
	leamer_robust = [True, False, False, False, False, False, False, False, False, False]

	# Create the figure
	fig = go.Figure()

	# Add bars
	fig.add_trace(go.Bar(
		y=variables,
		x=robust_pct,
		orientation='h',
		marker=dict(
			color=np.where(np.array(leamer_robust), 'rgba(16, 185, 129, 0.7)', 'rgba(59, 130, 246, 0.7)'),
			line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
		)
	))

	# Add vertical line at 0.95
	fig.add_shape(
		type='line',
		x0=0.95, y0=-1,
		x1=0.95, y1=10,
		line=dict(color='red', dash='dash')
	)

	# Add annotations
	fig.add_annotation(
		x=0.95, y=10.5,
		text="Sala-i-Martin's 95% Robustness Threshold",
		showarrow=False,
		font=dict(color="red")
	)

	fig.add_annotation(
		x=0.5, y=0,
		text="Only Initial Income passes Leamer's test",
		showarrow=True,
		arrowhead=1,
		ax=0,
		ay=-40,
		font=dict(color="#10B981")
	)

	fig.update_layout(
		title="Robustness of Growth Determinants (Illustrative Example)",
		xaxis_title="Fraction of Distribution on Same Side of Zero",
		yaxis_title="Variable",
		height=500
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    This illustrative example (based on Sala-i-Martin's findings) shows how different the conclusions can be:

    - Using Leamer's strict criteria, only initial income level would be considered a robust determinant of growth.

    - Using Sala-i-Martin's approach with a 95% threshold, several additional variables would be considered 
      robust, including life expectancy, education, investment rate, and others.

    This demonstrates why Sala-i-Martin developed his modified approach - Leamer's criteria were so stringent 
    that they failed to identify variables that seemed economically significant based on both theory and empirical 
    evidence.
    """)

	st.markdown("<div class='section-header'>Modern Developments</div>", unsafe_allow_html=True)

	st.markdown("""
    Since the development of these methods, several advancements and alternatives have emerged:

    1. **Bayesian Model Averaging (BMA)**: A more formal Bayesian approach that considers posterior 
       probability distributions over the model space.

    2. **WALS (Weighted-Average Least Squares)**: Combines Bayesian and frequentist approaches to model 
       averaging with specific prior distributions.

    3. **LASSO and Other Regularization Methods**: Modern machine learning approaches to variable selection 
       that penalize complexity.

    4. **Bayesian Extreme Bounds Analysis**: Modifications of EBA that incorporate prior distributions more 
       explicitly.

    Despite these advancements, the core insights from Leamer and Sala-i-Martin's approaches remain relevant, 
    and their methods continue to be used in applied research.
    """)

# Practical Applications
elif section == "Practical Applications":
	st.markdown("<div class='sub-header'>Practical Applications of Extreme Bounds Analysis</div>",
				unsafe_allow_html=True)

	st.markdown("""
    Extreme Bounds Analysis (EBA) has been applied to various fields to address model uncertainty and identify robust 
    relationships. This section explores some key applications and provides examples of how EBA has been used in practice.
    """)

	st.markdown("<div class='section-header'>Economic Growth Determinants</div>", unsafe_allow_html=True)

	st.markdown("""
    The most famous application of EBA is in identifying robust determinants of economic growth across countries. 
    This was the focus of both Levine and Renelt's (1992) paper using Leamer's approach and Sala-i-Martin's (1997) 
    "I Just Ran Two Million Regressions" paper.
    """)

	# Create a visualization of growth determinants
	determinants = [
		"Initial Income",
		"Education",
		"Investment Rate",
		"Rule of Law",
		"Inflation",
		"Government Consumption",
		"Trade Openness",
		"Political Stability",
		"Life Expectancy",
		"Population Growth"
	]

	# Simulated robustness values for Leamer and Sala-i-Martin methods
	leamer_robust = [True, False, True, False, False, False, False, False, False, False]
	sala_robust = [True, True, True, True, True, False, True, False, True, False]

	leamer_values = [0.99, 0.88, 0.96, 0.85, 0.78, 0.60, 0.89, 0.75, 0.87, 0.65]
	sala_values = [0.99, 0.97, 0.98, 0.96, 0.95, 0.88, 0.96, 0.93, 0.97, 0.83]

	# Create the figure
	fig = go.Figure()

	# Add bars for Sala-i-Martin values
	fig.add_trace(go.Bar(
		y=determinants,
		x=sala_values,
		orientation='h',
		name="Sala-i-Martin",
		marker_color='rgba(59, 130, 246, 0.7)'
	))

	# Add bars for Leamer values
	fig.add_trace(go.Bar(
		y=determinants,
		x=leamer_values,
		orientation='h',
		name="Leamer",
		marker_color='rgba(16, 185, 129, 0.7)'
	))

	# Add threshold lines
	fig.add_shape(
		type='line',
		x0=0.95, y0=-1,
		x1=0.95, y1=10,
		line=dict(color='rgba(59, 130, 246, 0.7)', dash='dash'),
		name="Sala-i-Martin Threshold"
	)

	fig.add_shape(
		type='line',
		x0=0.95, y0=-1,
		x1=0.95, y1=10,
		line=dict(color='rgba(16, 185, 129, 0.7)', dash='dash'),
		name="Leamer Threshold"
	)

	fig.update_layout(
		title="Robustness of Growth Determinants by Method (Illustrative Example)",
		xaxis_title="Robustness Measure",
		yaxis_title="Growth Determinant",
		height=500,
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates (with hypothetical data) how Leamer's and Sala-i-Martin's approaches 
    might evaluate the robustness of various determinants of economic growth. Notice how Leamer's approach 
    typically identifies fewer variables as robust.

    Some key findings from actual research on growth determinants:

    - **Initial income level** is consistently identified as robust (supporting convergence theory)
    - **Investment rate** is generally robust across methods
    - **Education measures** are typically robust in Sala-i-Martin's approach but not always in Leamer's
    - **Institutional quality** (rule of law, property rights) often emerges as robust in Sala-i-Martin's approach
    - **Policy variables** (inflation, government spending) show mixed results
    """)

	st.markdown("<div class='section-header'>Financial Development and Banking</div>", unsafe_allow_html=True)

	st.markdown("""
    EBA has been widely applied in banking and finance research to identify robust determinants of:

    - Banking crises and financial stability
    - Bank profitability and performance
    - Capital structure decisions
    - Financial development and its relationship to growth

    For example, Levine and Renelt (1992) examined the robustness of the relationship between financial 
    development and economic growth, finding that certain measures of financial depth are robustly correlated 
    with investment rates and growth.
    """)

	# Create a simple visual for financial determinants
	fin_determinants = [
		"Financial Depth",
		"Bank Concentration",
		"Capital Requirements",
		"Financial Openness",
		"Central Bank Independence",
		"Deposit Insurance"
	]

	# Create simulated data for various financial outcomes
	outcomes = ["Banking Crisis", "Bank Profitability", "Financial Development"]

	# Create a matrix of robustness (1 = robust, 0 = not robust)
	robustness = np.array([
		[0, 1, 1],  # Financial Depth
		[1, 1, 0],  # Bank Concentration
		[1, 0, 0],  # Capital Requirements
		[0, 0, 1],  # Financial Openness
		[1, 0, 0],  # Central Bank Independence
		[1, 0, 0]  # Deposit Insurance
	])

	# Create heatmap
	fig = px.imshow(
		robustness,
		labels=dict(x="Outcome Variable", y="Determinant", color="Robust"),
		x=outcomes,
		y=fin_determinants,
		color_continuous_scale=["white", "#3B82F6"],
		title="Robust Determinants in Banking & Finance (Illustrative Example)"
	)

	fig.update_layout(height=400)
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The heatmap above illustrates (with hypothetical data) how EBA might be used to identify robust 
    determinants of various financial outcomes. For example, deposit insurance might be robustly related 
    to banking crises but not to other outcomes.
    """)

	st.markdown("<div class='section-header'>Political Economy and Institutions</div>", unsafe_allow_html=True)

	st.markdown("""
    EBA has been applied to political economy questions such as:

    - Determinants of democracy and political stability
    - Effects of political institutions on economic outcomes
    - Corruption and its determinants
    - Political budget cycles

    Researchers have used EBA to examine whether variables like income levels, education, ethnic fractionalization, 
    and colonial history are robustly related to institutional quality.
    """)

	# Create an example for institutional determinants
	democracy_data = {
		"Variable": [
			"GDP per Capita",
			"Education",
			"Oil Resources",
			"Ethnic Fractionalization",
			"Colonial History",
			"Religion",
			"Trade Openness",
			"Income Inequality"
		],
		"Coefficient": [0.42, 0.38, -0.35, -0.28, -0.22, -0.18, 0.15, -0.12],
		"CDF(0)": [0.99, 0.98, 0.97, 0.94, 0.90, 0.87, 0.82, 0.73],
		"Robust": [True, True, True, False, False, False, False, False]
	}

	democracy_df = pd.DataFrame(democracy_data)

	# Create visual
	fig = go.Figure()

	# Add bars for coefficients
	fig.add_trace(go.Bar(
		x=democracy_df["Variable"],
		y=democracy_df["Coefficient"],
		name="Coefficient",
		marker_color=np.where(democracy_df["Coefficient"] > 0, 'rgba(16, 185, 129, 0.7)', 'rgba(239, 68, 68, 0.7)')
	))

	# Add line for CDF
	fig.add_trace(go.Scatter(
		x=democracy_df["Variable"],
		y=democracy_df["CDF(0)"],
		mode='lines+markers',
		name="CDF(0)",
		marker=dict(size=10),
		line=dict(color='rgba(59, 130, 246, 0.7)', width=3)
	))

	# Add threshold line for robustness
	fig.add_shape(
		type='line',
		x0=-0.5, y0=0.95,
		x1=7.5, y1=0.95,
		line=dict(color='red', dash='dash')
	)

	fig.update_layout(
		title="Determinants of Democracy (Illustrative Example)",
		xaxis_title="Variable",
		yaxis=dict(
			title="Coefficient",
			side="left"
		),
		yaxis2=dict(
			title="CDF(0)",
			side="right",
			overlaying="y",
			range=[0, 1]
		),
		height=500,
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above shows a hypothetical application of EBA to determinants of democracy. In this example, 
    income level, education, and oil resources would be considered robust determinants of democracy (though 
    oil has a negative relationship), while other factors would not meet the robustness criteria.
    """)

	st.markdown("<div class='section-header'>Health Economics and Public Health</div>", unsafe_allow_html=True)

	st.markdown("""
    EBA has been applied in health economics to identify robust determinants of:

    - Health outcomes and life expectancy
    - Healthcare expenditure
    - Healthcare system efficiency
    - Health behaviors and public health

    For example, researchers have used EBA to examine what factors robustly explain differences in healthcare 
    spending across countries or regions.
    """)

	# Create a visualization for health determinants
	health_determinants = [
		"Income",
		"Education",
		"Healthcare Access",
		"Lifestyle Factors",
		"Environmental Quality",
		"Healthcare System Type",
		"Social Support",
		"Public Health Spending"
	]

	# Create dummy data for two outcomes: life expectancy and healthcare expenditure
	life_exp_robust = [True, True, True, True, False, False, True, False]
	expenditure_robust = [True, False, False, False, False, True, False, True]

	# Create figure
	fig = go.Figure()

	# Add scatter points for life expectancy
	fig.add_trace(go.Scatter(
		x=health_determinants,
		y=[1 if r else 0 for r in life_exp_robust],
		mode='markers',
		marker=dict(
			size=20,
			symbol='circle',
			color=['#10B981' if r else '#d1d5db' for r in life_exp_robust],
			line=dict(width=1)
		),
		name="Life Expectancy"
	))

	# Add scatter points for healthcare expenditure
	fig.add_trace(go.Scatter(
		x=health_determinants,
		y=[0.5 if r else -0.5 for r in expenditure_robust],
		mode='markers',
		marker=dict(
			size=20,
			symbol='diamond',
			color=['#3B82F6' if r else '#d1d5db' for r in expenditure_robust],
			line=dict(width=1)
		),
		name="Healthcare Expenditure"
	))

	fig.update_layout(
		title="Robust Determinants of Health Outcomes (Illustrative Example)",
		xaxis=dict(
			title="Determinant",
			tickangle=45
		),
		yaxis=dict(
			showticklabels=False,
			range=[-1, 1.5],
			zeroline=False
		),
		height=400,
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		),
		shapes=[
			dict(
				type='line',
				x0=-0.5, y0=0.25,
				x1=7.5, y1=0.25,
				line=dict(color='rgba(16, 185, 129, 0.3)', width=50)
			),
			dict(
				type='line',
				x0=-0.5, y0=-0.25,
				x1=7.5, y1=-0.25,
				line=dict(color='rgba(59, 130, 246, 0.3)', width=50)
			)
		]
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates (with hypothetical data) how EBA might identify different robust determinants 
    for different health outcomes. For example, income is robustly related to both life expectancy and healthcare 
    expenditure, while education is only robustly related to life expectancy.
    """)

	st.markdown("<div class='section-header'>Environmental Economics</div>", unsafe_allow_html=True)

	st.markdown("""
    EBA has been applied to environmental questions such as:

    - Determinants of carbon emissions and environmental degradation
    - Environmental Kuznets Curve (EKC) hypothesis testing
    - Effectiveness of environmental policies
    - Climate change impacts and adaptation

    For example, researchers have used EBA to test whether the EKC relationship (an inverted U-shaped relationship 
    between income and pollution) is robust to different model specifications.
    """)

	# Create a visual for Environmental Kuznets Curve testing
	# Generate simulated data for different specifications of the EKC model
	np.random.seed(42)

	income_levels = np.linspace(0, 10, 100)

	# Generate multiple models with slightly different shapes
	models = []
	for i in range(10):
		a = np.random.uniform(0.8, 1.2)
		b = np.random.uniform(0.8, 1.2) * -0.2
		c = np.random.uniform(0.8, 1.2) * 0.01
		pollution = a + b * income_levels + c * income_levels ** 2
		models.append(pollution)

	# Create the figure
	fig = go.Figure()

	# Add lines for different model specifications
	for i, model in enumerate(models):
		fig.add_trace(go.Scatter(
			x=income_levels,
			y=model,
			mode='lines',
			line=dict(color='rgba(59, 130, 246, 0.3)'),
			showlegend=False
		))

	# Add a thicker line for the average model
	avg_model = np.mean(models, axis=0)
	fig.add_trace(go.Scatter(
		x=income_levels,
		y=avg_model,
		mode='lines',
		line=dict(color='#3B82F6', width=3),
		name="Average Relationship"
	))

	# Add vertical line at turning point
	turning_point = income_levels[np.argmax(avg_model)]
	fig.add_shape(
		type='line',
		x0=turning_point, y0=0,
		x1=turning_point, y1=1.5,
		line=dict(color='red', dash='dash')
	)

	fig.add_annotation(
		x=turning_point,
		y=1.5,
		text="Turning Point",
		showarrow=True,
		arrowhead=1,
		ax=40,
		ay=-40
	)

	fig.update_layout(
		title="Testing the Environmental Kuznets Curve with EBA",
		xaxis_title="Income Level",
		yaxis_title="Pollution Level",
		height=400
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates how EBA might be used to test the robustness of the Environmental Kuznets Curve 
    hypothesis. By running many regressions with different model specifications, researchers can determine whether 
    the inverted U-shaped relationship is robust and where the turning point occurs.
    """)

	st.markdown("<div class='section-header'>Implementation Steps</div>", unsafe_allow_html=True)

	st.markdown("""
    If you're planning to implement EBA in your own research, here are the key steps to follow:
    """)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
        ### For Leamer's EBA:

        1. **Define variable sets**:
           - Variable of interest (M)
           - Free/fixed variables (F)
           - Doubtful variables (Z)

        2. **Run multiple regressions**:
           - For each Z combination
           - Record coefficient and SE for M

        3. **Find extreme bounds**:
           - Lower bound = min(β - 2×SE)
           - Upper bound = max(β + 2×SE)

        4. **Apply robustness criteria**:
           - Check if bounds have same sign
           - Check if bounds are significant
        """)

	with col2:
		st.markdown("""
        ### For Sala-i-Martin's Approach:

        1. **Define variable sets** (same as Leamer)

        2. **Run multiple regressions**:
           - For each Z combination
           - Record coefficient for M

        3. **Analyze coefficient distribution**:
           - Calculate CDF(0)
           - Or use weighted approach

        4. **Apply robustness criteria**:
           - Check if CDF(0) ≥ 0.95 (or ≤ 0.05)
           - Report fraction of distribution
        """)

	st.markdown("""
    ### Important Practical Considerations:

    1. **Computational Requirements**: For a large set of Z variables, it may be infeasible to run all possible 
       combinations. Researchers often use:
       - Random sampling of models
       - Focusing on combinations of theoretical interest
       - Advanced computational techniques

    2. **Multicollinearity**: High correlation among explanatory variables can affect results. Consider:
       - Examining correlation matrices
       - Using principal components
       - Grouping similar variables

    3. **Interpretation**: Remember that EBA tests robustness, not causality. Robust relationships still 
       require theoretical justification and other causal identification strategies.
    """)

	st.markdown("<div class='section-header'>Research Examples</div>", unsafe_allow_html=True)

	st.markdown("""
    Here are some influential papers that have applied EBA in different fields:

    1. **Economic Growth**:
       - Levine & Renelt (1992): "A Sensitivity Analysis of Cross-Country Growth Regressions"
       - Sala-i-Martin (1997): "I Just Ran Two Million Regressions"
       - Fernandez, Ley & Steel (2001): "Model Uncertainty in Cross-Country Growth Regressions"

    2. **Finance and Banking**:
       - Beck, Demirgüç-Kunt & Levine (2006): "Bank Concentration, Competition, and Crises"
       - Demirgüç-Kunt & Detragiache (1998): "The Determinants of Banking Crises in Developing and Developed Countries"

    3. **Political Economy**:
       - Gassebner, Lamla & Vreeland (2013): "Extreme Bounds of Democracy"
       - Dreher, Sturm & de Haan (2010): "When is a Central Bank Governor Replaced?"

    4. **Health Economics**:
       - Carmignani, Shankar & Tang (2018): "Identifying the robust economic, geographical and political determinants of FDI"
       - Hartwig & Sturm (2018): "Testing the Grossman model of medical spending determinants with macroeconomic panel data"

    5. **Environmental Economics**:
       - Gassebner, Lamla & Sturm (2011): "Determinants of pollution: what do we really know?"
       - Mehmet & Gülden (2016): "The relationship between CO2 emissions, energy consumption and economic growth"
    """)

# Limitations and Critiques
elif section == "Limitations and Critiques":
	st.markdown("<div class='sub-header'>Limitations and Critiques of Extreme Bounds Analysis</div>",
				unsafe_allow_html=True)

	st.markdown("""
    While Extreme Bounds Analysis (EBA) provides valuable tools for addressing model uncertainty, 
    it has faced various criticisms and has several important limitations. This section examines these 
    issues and discusses alternative approaches.
    """)

	st.markdown("<div class='section-header'>Theoretical Limitations</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>1. Equal Treatment of All Model Specifications</b><br>
    Both Leamer's and Sala-i-Martin's approaches (especially the unweighted version) can give equal weight to 
    theoretically implausible or misspecified models. This may distort the assessment of robustness.
    </div>
    """, unsafe_allow_html=True)

	# Create a visualization showing model plausibility
	np.random.seed(42)

	# Generate some model specifications with varying plausibility
	models = ['Model ' + str(i + 1) for i in range(10)]
	plausibility = np.random.uniform(0.1, 1.0, 10)
	plausibility.sort()
	coefficients = np.random.normal(0.3, 0.1, 10) * plausibility  # Coefficients somewhat related to plausibility

	fig = go.Figure()

	# Add bars for plausibility
	fig.add_trace(go.Bar(
		x=models,
		y=plausibility,
		name="Theoretical Plausibility",
		marker_color='rgba(59, 130, 246, 0.7)'
	))

	# Add line for coefficients
	fig.add_trace(go.Scatter(
		x=models,
		y=coefficients,
		mode='lines+markers',
		name="Coefficient Estimate",
		marker=dict(size=10),
		line=dict(color='#EF4444', width=3)
	))

	fig.update_layout(
		title="Theoretical Plausibility vs. Coefficient Estimates Across Models",
		xaxis_title="Model Specification",
		yaxis_title="Plausibility Score",
		height=400,
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates the issue of treating all models equally. Some model specifications may have 
    low theoretical plausibility but still influence the robustness assessment in traditional EBA. 
    Sala-i-Martin's weighted approach partly addresses this issue by giving more weight to models with better fit.
    """)

	st.markdown("""
    <div class="concept-box">
    <b>2. Focus on Sign and Statistical Significance</b><br>
    EBA primarily focuses on whether coefficients maintain their sign and statistical significance 
    across specifications, potentially overlooking the economic significance of effect sizes. A variable 
    might be statistically robust but have effects too small to be economically meaningful.
    </div>
    """, unsafe_allow_html=True)

	# Create a visualization showing statistical vs. economic significance
	np.random.seed(123)

	# Generate some coefficient estimates with different sample sizes
	sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
	coefficients = np.random.normal(0.05, 0.01, len(sample_sizes))  # Small but stable coefficient
	std_errors = 0.2 / np.sqrt(sample_sizes)  # Standard errors decrease with sample size
	t_stats = coefficients / std_errors

	fig = go.Figure()

	# Add bars for t-statistics
	fig.add_trace(go.Bar(
		x=[str(s) for s in sample_sizes],
		y=t_stats,
		name="t-statistic",
		marker_color=np.where(t_stats > 1.96, 'rgba(16, 185, 129, 0.7)', 'rgba(239, 68, 68, 0.7)')
	))

	# Add line for coefficients
	fig.add_trace(go.Scatter(
		x=[str(s) for s in sample_sizes],
		y=coefficients,
		mode='lines+markers',
		name="Coefficient (Effect Size)",
		marker=dict(size=10),
		line=dict(color='#3B82F6', width=3)
	))

	# Add horizontal line for significance threshold
	fig.add_shape(
		type='line',
		x0=-0.5, y0=1.96,
		x1=7.5, y1=1.96,
		line=dict(color='red', dash='dash')
	)

	fig.update_layout(
		title="Statistical vs. Economic Significance",
		xaxis_title="Sample Size",
		yaxis=dict(
			title="t-statistic",
			side="left"
		),
		yaxis2=dict(
			title="Coefficient Value",
			side="right",
			overlaying="y",
			range=[0, 0.1]
		),
		height=450,
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates how a variable can be statistically significant (especially with large sample sizes) 
    but have a small effect size that may lack economic significance. Traditional EBA would consider this variable 
    robust if it consistently shows statistical significance, even if its actual impact is minimal.
    """)

	st.markdown("""
    <div class="concept-box">
    <b>3. Sensitivity to Outliers</b><br>
    Leamer's approach, in particular, is highly sensitive to outliers since it focuses on extreme bounds. 
    A single unusual model specification can dramatically affect the conclusions about robustness.
    </div>
    """, unsafe_allow_html=True)

	# Create a visualization showing sensitivity to outliers
	np.random.seed(456)

	# Generate coefficient estimates across model specifications
	coefs = np.random.normal(0.5, 0.1, 30)
	std_errors = np.random.uniform(0.1, 0.15, 30)
	lower_bounds = coefs - 2 * std_errors
	upper_bounds = coefs + 2 * std_errors

	# Add one outlier
	coefs = np.append(coefs, -0.3)
	std_errors = np.append(std_errors, 0.15)
	lower_bounds = np.append(lower_bounds, -0.3 - 2 * 0.15)
	upper_bounds = np.append(upper_bounds, -0.3 + 2 * 0.15)

	# Calculate extreme bounds
	extreme_lower_without_outlier = min(lower_bounds[:-1])
	extreme_upper_without_outlier = max(upper_bounds[:-1])
	extreme_lower_with_outlier = min(lower_bounds)
	extreme_upper_with_outlier = max(upper_bounds)

	# Create the figure
	fig = go.Figure()

	# Add confidence intervals for each model (without the outlier)
	for i in range(30):
		fig.add_trace(go.Scatter(
			x=[i, i],
			y=[lower_bounds[i], upper_bounds[i]],
			mode='lines',
			line=dict(color='rgba(59, 130, 246, 0.3)', width=1),
			showlegend=False
		))

	# Add the outlier
	fig.add_trace(go.Scatter(
		x=[30, 30],
		y=[lower_bounds[30], upper_bounds[30]],
		mode='lines',
		line=dict(color='rgba(239, 68, 68, 0.8)', width=3),
		name="Outlier Model"
	))

	# Add coefficient points (without the outlier)
	fig.add_trace(go.Scatter(
		x=list(range(30)),
		y=coefs[:30],
		mode='markers',
		marker=dict(color='#3B82F6', size=6),
		name='β Estimates'
	))

	# Add the outlier point
	fig.add_trace(go.Scatter(
		x=[30],
		y=[coefs[30]],
		mode='markers',
		marker=dict(color='#EF4444', size=10),
		showlegend=False
	))

	# Add extreme bounds without outlier
	fig.add_shape(
		type='line',
		x0=-1, y0=extreme_lower_without_outlier,
		x1=31, y1=extreme_lower_without_outlier,
		line=dict(color='#3B82F6', width=2, dash='dash')
	)

	fig.add_shape(
		type='line',
		x0=-1, y0=extreme_upper_without_outlier,
		x1=31, y1=extreme_upper_without_outlier,
		line=dict(color='#3B82F6', width=2, dash='dash')
	)

	# Add extreme bounds with outlier
	fig.add_shape(
		type='line',
		x0=-1, y0=extreme_lower_with_outlier,
		x1=31, y1=extreme_lower_with_outlier,
		line=dict(color='#EF4444', width=2, dash='dash')
	)

	# Add zero line
	fig.add_shape(
		type='line',
		x0=-1, y0=0,
		x1=31, y1=0,
		line=dict(color='black', dash='dot')
	)

	# Add annotations
	fig.add_annotation(
		x=28, y=extreme_lower_without_outlier,
		text="Extreme Bounds Without Outlier",
		showarrow=True,
		arrowhead=1,
		ax=40,
		ay=20
	)

	fig.add_annotation(
		x=28, y=extreme_lower_with_outlier,
		text="Extreme Bounds With Outlier",
		showarrow=True,
		arrowhead=1,
		ax=40,
		ay=20
	)

	fig.update_layout(
		title="Sensitivity to Outliers in Leamer's EBA",
		xaxis_title="Model Specification",
		yaxis_title="Coefficient of Variable (β)",
		height=500,
		xaxis=dict(showticklabels=False)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above demonstrates how a single outlier model specification can dramatically change the extreme bounds 
    in Leamer's approach. Without the outlier, the variable might be considered robust (both extreme bounds are positive), 
    but with the outlier, the lower extreme bound becomes negative, leading to a conclusion that the variable is not robust.

    Sala-i-Martin's approach is less sensitive to outliers since it examines the entire distribution of coefficients, 
    but it can still be affected by extreme values.
    """)

	st.markdown("""
    <div class="concept-box">
    <b>4. Ignoring Theoretical Relationships Between Variables</b><br>
    EBA treats each model specification as independent, potentially ignoring theoretical relationships 
    between variables. This can lead to including models that violate basic theoretical constraints.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("<div class='section-header'>Practical Limitations</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>1. Computational Demands</b><br>
    With k potential control variables, there are 2^k possible model combinations, which can be computationally 
    intensive. This often necessitates sampling from the model space rather than exhaustive testing.
    </div>
    """, unsafe_allow_html=True)

	# Create a visualization of computational complexity
	variables = list(range(1, 21))
	combinations = [2 ** v for v in variables]

	fig = go.Figure()

	# Add bar for combinations
	fig.add_trace(go.Bar(
		x=[str(v) for v in variables],
		y=combinations,
		marker_color='rgba(59, 130, 246, 0.7)'
	))

	# Use log scale for y-axis
	fig.update_layout(
		title="Computational Complexity: Number of Possible Model Combinations",
		xaxis_title="Number of Variables",
		yaxis_title="Number of Possible Combinations",
		yaxis_type="log",
		height=400
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates the exponential growth in the number of possible model combinations as the 
    number of variables increases. With just 20 potential control variables, there are over 1 million possible 
    model specifications to test, which can be computationally prohibitive.
    """)

	st.markdown("""
    <div class="concept-box">
    <b>2. Multicollinearity Issues</b><br>
    High correlation among explanatory variables can lead to unstable coefficient estimates across different 
    specifications. This can make EBA results less reliable, as coefficient signs might flip due to 
    multicollinearity rather than true model uncertainty.
    </div>
    """, unsafe_allow_html=True)

	# Create a visualization of multicollinearity effects
	np.random.seed(789)

	# Generate correlated variables
	x1 = np.random.normal(0, 1, 100)
	x2 = 0.9 * x1 + np.random.normal(0, 0.2, 100)  # Highly correlated with x1

	# Generate dependent variable
	y = 1 + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 1, 100)

	# Run three regressions:
	# 1. y ~ x1
	# 2. y ~ x2
	# 3. y ~ x1 + x2

	# Simplified results (manually computed for illustration)
	models = ['y ~ x1', 'y ~ x2', 'y ~ x1 + x2']
	x1_coefs = [0.8, 0.0, 0.6]  # Coefficient of x1 in each model
	x2_coefs = [0.0, 0.7, 0.2]  # Coefficient of x2 in each model

	# Create the figure
	fig = go.Figure()

	# Add bars for x1 coefficients
	fig.add_trace(go.Bar(
		x=models,
		y=x1_coefs,
		name="Coefficient of x1",
		marker_color='rgba(16, 185, 129, 0.7)'
	))

	# Add bars for x2 coefficients
	fig.add_trace(go.Bar(
		x=models,
		y=x2_coefs,
		name="Coefficient of x2",
		marker_color='rgba(59, 130, 246, 0.7)'
	))

	fig.update_layout(
		title="Effect of Multicollinearity on Coefficient Estimates",
		xaxis_title="Model Specification",
		yaxis_title="Coefficient Value",
		height=400,
		barmode='group',
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates how multicollinearity can affect coefficient estimates across different model 
    specifications. When highly correlated variables (x1 and x2) are included separately, they both show strong 
    effects. However, when included together, their coefficients diminish due to multicollinearity. This instability 
    can make EBA results less reliable.
    """)

	st.markdown("""
    <div class="concept-box">
    <b>3. Lack of Consideration for Model Fit</b><br>
    Leamer's approach doesn't consider how well different models fit the data. While Sala-i-Martin's weighted 
    approach addresses this somewhat, the weighting scheme may still not fully capture model quality.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>4. Potential for Data Mining</b><br>
    The process of running many regressions and selecting results based on certain criteria can lead to 
    data mining concerns, where researchers might focus on specifications that yield desired results.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("<div class='section-header'>Philosophical and Methodological Critiques</div>", unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>1. Confusing Robustness with Causality</b><br>
    EBA tests whether relationships are robust to model specification, not whether they are causal. 
    A robust correlation does not imply causation, and this distinction is sometimes overlooked in applications of EBA.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>2. Tension Between Theory and Data-Driven Approaches</b><br>
    EBA is fundamentally a data-driven approach to addressing model uncertainty, which may conflict with 
    the emphasis on theory-driven model specification in some fields. Critics argue that theoretical considerations 
    should guide model selection more than statistical robustness.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>3. Ad Hoc Nature of Robustness Criteria</b><br>
    The criteria for determining robustness (e.g., 95% of distribution on one side of zero) are somewhat 
    arbitrary. Different thresholds could lead to different conclusions about which variables are robust.
    </div>
    """, unsafe_allow_html=True)

	# Create a visualization of varying thresholds
	np.random.seed(42)

	# Generate coefficient distribution
	coefs = np.random.normal(0.2, 0.15, 1000)

	# Calculate robustness under different thresholds
	thresholds = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
	robust = [(coefs > 0).mean() >= t for t in thresholds]

	# Create the figure
	fig = go.Figure()

	# Add histogram of coefficients
	fig.add_trace(go.Histogram(
		x=coefs,
		nbinsx=30,
		marker_color='rgba(59, 130, 246, 0.7)',
		name="Coefficient Distribution"
	))

	# Add vertical line at zero
	fig.add_shape(
		type='line',
		x0=0, y0=0,
		x1=0, y1=120,
		line=dict(color='red', dash='dash')
	)

	# Calculate percentage positive
	pos_pct = (coefs > 0).mean() * 100

	# Add annotation
	fig.add_annotation(
		x=0.4, y=100,
		text=f"{pos_pct:.1f}% of distribution > 0",
		showarrow=False,
		font=dict(size=14)
	)

	fig.update_layout(
		title="Effect of Different Robustness Thresholds",
		xaxis_title="Coefficient Value",
		yaxis_title="Frequency",
		height=400
	)

	# Add threshold lines
	thresholds_text = []
	for i, t in enumerate(thresholds):
		color = 'rgba(16, 185, 129, 0.7)' if robust[i] else 'rgba(239, 68, 68, 0.7)'
		thresholds_text.append(
			f"Threshold {t * 100}%: {'Robust' if robust[i] else 'Not Robust'}"
		)

	fig.add_annotation(
		x=0.4, y=80,
		text="<br>".join(thresholds_text),
		showarrow=False,
		align="left",
		bgcolor="rgba(255, 255, 255, 0.8)",
		font=dict(size=12)
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The figure above illustrates how different robustness thresholds can lead to different conclusions. 
    In this example, the variable would be considered robust with a 90% threshold but not with a 95% threshold. 
    The choice of threshold is somewhat arbitrary and can significantly affect the results.
    """)

	st.markdown("<div class='section-header'>Alternative Approaches</div>", unsafe_allow_html=True)

	st.markdown("""
    Several alternative approaches have been developed to address the limitations of traditional EBA:
    """)

	st.markdown("""
    <div class="concept-box">
    <b>1. Bayesian Model Averaging (BMA)</b><br>
    BMA provides a more formal Bayesian framework for dealing with model uncertainty. Instead of focusing on 
    extreme bounds or simple distributions, BMA calculates the posterior probability of each model and uses these 
    to weight the coefficient estimates.

    The posterior mean of a coefficient β is calculated as:
    </div>
    """, unsafe_allow_html=True)

	st.latex(r'''
    E(\beta|D) = \sum_{j=1}^{2^K} P(M_j|D) \cdot E(\beta|M_j, D)
    ''')

	st.markdown("""
    Where:
    * $P(M_j|D)$ is the posterior probability of model $j$ given the data
    * $E(\beta|M_j, D)$ is the expected value of $\beta$ in model $j$
    """)

	st.markdown("""
    <div class="concept-box">
    <b>2. WALS (Weighted-Average Least Squares)</b><br>
    WALS is a hybrid approach that combines elements of Bayesian and frequentist methods. It uses a specific 
    semi-orthogonal transformation of the auxiliary regressors and applies a Bayesian perspective with 
    a particular prior distribution.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
    <b>3. Machine Learning Approaches</b><br>
    Modern machine learning methods like LASSO, Ridge regression, and Elastic Net provide alternatives for 
    variable selection in the presence of model uncertainty. These methods use regularization to shrink 
    coefficients toward zero, effectively performing variable selection.
    </div>
    """, unsafe_allow_html=True)

	# Create a visual comparison of approaches
	approaches = [
		"Leamer's EBA",
		"Sala-i-Martin's EBA",
		"Bayesian Model Averaging",
		"WALS",
		"LASSO/Ridge Regression"
	]

	criteria = [
		"Theoretical Foundation",
		"Computational Complexity",
		"Handles Multicollinearity",
		"Considers Model Fit",
		"Provides Uncertainty Measures"
	]

	# Create a score matrix (hypothetical scores)
	scores = np.array([
		[4, 3, 5, 4, 3],  # Leamer's EBA
		[3, 2, 3, 4, 4],  # Sala-i-Martin's EBA
		[5, 1, 4, 5, 5],  # BMA
		[4, 3, 5, 4, 4],  # WALS
		[3, 5, 5, 3, 3]  # LASSO/Ridge
	])

	# Create heatmap
	fig = px.imshow(
		scores,
		labels=dict(x="Criterion", y="Approach", color="Score (1-5)"),
		x=criteria,
		y=approaches,
		color_continuous_scale="Viridis",
		text_auto=True
	)

	fig.update_layout(
		title="Comparison of Approaches to Model Uncertainty",
		height=400
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    The heatmap above provides a hypothetical comparison of different approaches to addressing model uncertainty 
    across various criteria. Each approach has strengths and weaknesses, and the choice depends on the specific 
    research context and objectives.
    """)

	st.markdown("<div class='section-header'>Best Practices and Recommendations</div>", unsafe_allow_html=True)

	st.markdown("""
    Despite its limitations, EBA remains a useful tool for addressing model uncertainty when applied appropriately. 
    Here are some best practices:

    1. **Use Multiple Approaches**: Apply both traditional EBA and alternative methods like BMA to check 
       for consistency in results.

    2. **Consider Theoretical Foundations**: Use theory to guide which variables are included in the free set 
       and which combinations of doubtful variables are most plausible.

    3. **Examine Economic Significance**: Look beyond statistical significance to assess whether the 
       magnitude of effects is economically meaningful.

    4. **Be Transparent About Specification Choices**: Clearly document which models were estimated and how 
       robustness criteria were applied.

    5. **Use Weighted Approaches**: When possible, use approaches that weight models by their likelihood or fit.

    6. **Address Multicollinearity**: Examine correlations among explanatory variables and consider how 
       multicollinearity might affect EBA results.

    7. **Supplement with Causal Identification Strategies**: Remember that robustness does not imply causality; 
       use EBA alongside methods that address causal identification.
    """)

	st.markdown("<div class='section-header'>Conclusion</div>", unsafe_allow_html=True)

	st.markdown("""
    Extreme Bounds Analysis, in both its original Leamer form and Sala-i-Martin's modification, provides valuable 
    tools for addressing model uncertainty in empirical research. While these approaches have important limitations 
    and have faced various critiques, they remain useful when applied carefully and with an understanding of their 
    constraints.

    The field has evolved with more sophisticated approaches like Bayesian Model Averaging, but the fundamental 
    insights from EBA about the importance of testing the robustness of relationships across model specifications 
    remain relevant. By combining EBA with other methods and grounding analysis in sound theory, researchers can 
    enhance the credibility and reliability of their empirical findings.
    """)

# Add a footer
st.markdown("""
<div class="footnote">
Created by Dr Roudane Merwan | Extreme Bounds Analysis Theoretical Guide | For educational purposes only
</div>
""", unsafe_allow_html=True)
