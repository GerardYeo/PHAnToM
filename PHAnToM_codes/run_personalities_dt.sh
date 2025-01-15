MODEL_NAME="Llama-2-7b-chat-hf"
BATCH_SIZE=8

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=7 \
    python3 src/personality/dt_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

CUDA_VISIBLE_DEVICES=7 \
python3 src/personality/dt_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

MODEL_NAME="mistral-instruct"
BATCH_SIZE=8

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=7 \
    python3 src/personality/dt_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

CUDA_VISIBLE_DEVICES=7 \
python3 src/personality/dt_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

MODEL_NAME="zephyr-7b-beta"
BATCH_SIZE=8

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=7 \
    python3 src/personality/dt_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

CUDA_VISIBLE_DEVICES=7 \
python3 src/personality/dt_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

MODEL_NAME="falcon-7b-instruct"
BATCH_SIZE=8

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=7 \
    python3 src/personality/dt_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

CUDA_VISIBLE_DEVICES=7 \
python3 src/personality/dt_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

MODEL_NAME="gpt-3.5-turbo-1106"
BATCH_SIZE=8

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=7 \
    python3 src/personality/dt_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

CUDA_VISIBLE_DEVICES=7 \
python3 src/personality/dt_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE