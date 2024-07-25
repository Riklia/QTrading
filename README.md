# Financial Decision-Making using Deep Reinforcement Learning

Table of Content

- [Overview](#overview)
- [Introduction](#introduction)
- [Key features](#key-features-of-the-project)
- [Used technologies and algorithms](#technologies-used)
- [Installation](#introduction)
- [Usage](#usage)

## Overview

This project aims to develop an automated trading system for financial markets utilizing deep reinforcement learning, specifically employing the Double Deep Q-Learning (DDQN) algorithm. The system is designed to optimize trading strategies, minimize risks, and increase profits in highly volatile and unpredictable financial markets.

## Introduction

Financial markets are characterized by high volatility and unpredictability, presenting significant challenges for traders and investors. Automated trading systems based on machine learning algorithms have become increasingly relevant as they can optimize trading strategies, minimize risks, and increase profits.

The Double Deep Q-Learning (DDQN) algorithm, an advanced method in reinforcement learning, uses two neural networks to improve decision-making accuracy by reducing state overestimation.

## Key features of the project
* Utilizes DDQN to make buy or sell decisions for various assets.  
* Dynamically adjusts strategies based on real-time market changes.
* Easy to configure  
* Can be reused to make predictions not only for bitcoin, but for other currency.  

## Technologies Used
* Programming Languages: Python
* Libraries: TensorFlow, Keras, NumPy, Pandas
* Algorithms: Double Deep Q-Learning (DDQN)
* Data Sources: Real-time financial market data APIs

## Installation
Clone the repository:  

```bash
git clone https://github.com/Riklia/QTrading.git
```
Navigate to the project directory:  
```bash
cd QTrading
```
Install the required dependencies:  
```bash
pip install -r requirements.txt
```

## Usage 
Update the data source settings in `config.py`. Set train or test (inference) mode and other configurations. Run `python main.py`.
