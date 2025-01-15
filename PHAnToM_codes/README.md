# Psychology-of-Reasoning-in-LLMs

### Setting up
* See `fantom_eval`, install all the required packages
* Llama is gated, so ask Hf/Meta for access first (almost instant)
* Create a `data` folder with the following file structure

├── data<br>
│   ├── fantom<br>
│   │   ├── fantom_v1.json<br>
│   ├── empathy<br>
│   │   ├── messages.csv<br>
│   ├── personality<br>
│   │   ├── mpi_120.csv<br>

Fantom autodownloads, the other 2 CSVs need to manually download from Resources linked below.


### Running the model
Customize your experiments accordingly within `run_personalities.sh`. 

### Resources
Theory-of-Mind Reasoning Dataset: <br>
https://github.com/skywalker023/fantom/tree/main<br>

<br>

ToM Challenges Dataset<br>
https://github.com/xiaomeng-ma/ToMChallenges<br>
Download `Sally-Anne_prompt.csv` and `Smarties_prompt.csv`

<br>

Personality Prompting & MPI Dataset<br>
https://github.com/jianggy/MPI/tree/main<br>
Download `mpi_120.csv`

<br>

Empathatic Reactions Dataset<br>
https://github.com/wwbp/empathic_reactions/tree/master<br>
Download `messages.csv`