MODEL_NAME="Llama-2-7b-chat-hf"
BATCH_SIZE=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
sudo /home/XXXX-3/anaconda3/envs/py310/bin/python \
src/reasoning/tomi_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 \
    src/reasoning/tomi_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

MODEL_NAME="mistral-instruct"
BATCH_SIZE=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 \
src/reasoning/tomi_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 \
    src/reasoning/tomi_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

MODEL_NAME="zephyr-7b-beta"
BATCH_SIZE=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 \
src/reasoning/tomi_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 \
    src/reasoning/tomi_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

MODEL_NAME="falcon-7b-instruct"
BATCH_SIZE=12

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 \
src/reasoning/tomi_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 \
    src/reasoning/tomi_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done