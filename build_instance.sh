export IMAGE_FAMILY="pytorch-1-3-cu100-notebooks"
export ZONE="us-west1-b"
export INSTANCE_NAME="sonumator-classify-all-3"
export INSTANCE_TYPE="n1-standard-16"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image=sonumator-classify \
        --maintenance-policy=TERMINATE \
        --machine-type=$INSTANCE_TYPE \
        --preemptible