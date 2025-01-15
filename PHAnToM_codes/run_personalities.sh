MODEL_NAME="gpt-3.5-turbo-1106"
BATCH_SIZE=6

P2_SOURCE="p2_ours"
personality_list="agreeableness openness conscientiousness extraversion neuroticism task-specific narcissism machiavellianism psychopathy"
for personality in $personality_list; do 
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
    sudo /home/XXXX-3/anaconda3/envs/py310/bin/python \
    src/personality/mpi_eval.py \
    --model $MODEL_NAME \
    --personality $personality \
    --batch-size $BATCH_SIZE \
    --p2_source $P2_SOURCE
done

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
sudo /home/XXXX-3/anaconda3/envs/py310/bin/python \
src/personality/mpi_eval.py \
--model $MODEL_NAME \
--batch-size $BATCH_SIZE
