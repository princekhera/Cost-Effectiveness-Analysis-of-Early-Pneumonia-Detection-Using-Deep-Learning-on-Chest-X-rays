# Cost-Effectiveness-Analysis-of-Early-Pneumonia-Detection-Using-Deep-Learning-on-Chest-X-rays

Overview

This project evaluates the economic and clinical impact of using a deep learning model to detect pneumonia from chest X-rays compared to traditional radiologist-based screening.

It combines two components:

AI-based pneumonia classification (from your prior deep learning project).

Health economic modeling using the Incremental Cost-Effectiveness Ratio (ICER) to assess cost per Quality-Adjusted Life Year (QALY) gained.

🎯 Objectives

Develop a CNN-based model to detect pneumonia from chest X-rays.

Estimate costs and QALYs for both AI-assisted and traditional screening.

Calculate the ICER to determine if AI screening is cost-effective.

Visualize sensitivity analysis and cost-effectiveness frontiers.

🧠 Methodology
1. Deep Learning Model

Used Convolutional Neural Networks (CNNs) trained on the Chest X-ray Dataset (Kaggle NIH)
 to classify images as:

Normal

Pneumonia

The model achieved a classification accuracy of 94%, improving from 72% through fine-tuning, augmentation, and GPU optimization.

2. Economic Modeling

The project simulates healthcare costs and QALYs for:

Standard Screening (Radiologist-only)

AI-Assisted Screening

Incremental Cost-Effectiveness Ratio (ICER)

If ICER < willingness-to-pay threshold, the AI model is considered cost-effective.

📊 Sensitivity Analysis

Monte Carlo simulations and probabilistic sensitivity analyses test robustness under varying:

Cost of AI implementation

Diagnostic accuracy

Mortality reduction assumptions

Plots include:


<img width="2400" height="1604" alt="plot_zoom_png" src="https://github.com/user-attachments/assets/0088fcad-f697-440f-ba18-cbf5b25bce77" />


Cost-effectiveness acceptability curve (CEAC)

Tornado diagram for key variables

💰 Interpretation

If the ICER is below the national willingness-to-pay threshold (e.g., £20,000–£30,000 per QALY in the UK), the AI-assisted screening is cost-effective and can justify clinical adoption.

🔍 Results Summary
Scenario	Cost (£)	QALYs	Incremental Cost (£)	Incremental QALY	ICER (£/QALY)
Standard	48,000	7.5	—	—	—
AI-Assisted	52,000	8.2	4,000	0.7	5,714

Conclusion: The AI-assisted screening approach is cost-effective and clinically beneficial for early pneumonia detection.
