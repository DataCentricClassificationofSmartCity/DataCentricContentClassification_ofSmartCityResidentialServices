import kfp
import kfp.dsl as dsl

@dsl.pipeline(
    name='NLP main pipeline',
    description='A simple pipeline to run only 1 NLP training'
)
def nlp_pipeline():
    datasetVolume = dsl.VolumeOp(
        name="dataset-volume",
        resource_name="nlp-sd-dataset",
        modes=dsl.VOLUME_MODE_RWO,
        size="1Gi",
    )

    outputVolume = dsl.VolumeOp(
        name="output-volume",
        resource_name="nlp-sd-output",
        size="5Gi",
        modes=dsl.VOLUME_MODE_RWO,
    )

    getDatasets = dsl.ContainerOp(
        name="download Data from GCloud",
        image="google/cloud-sdk:alpine",
        command=['gsutil', 'cp', '-r', 'gs://nlp-sd/datasets/datasets.7z', "/workspace/nlp-smart-dispatching/datasets"],

        pvolumes = {
            "/workspace/nlp-smart-dispatching/datasets": datasetVolume.volume,
        }
    )

    extractData = dsl.ContainerOp(
        name="extract Data",
        image="delitescere/7z",
        command = ["7z", "x", "-o/workspace/nlp-smart-dispatching", "/workspace/nlp-smart-dispatching/datasets/datasets.7z"],

        pvolumes = {
            "/workspace/nlp-smart-dispatching/datasets": datasetVolume.volume,
        }
    )

    extractData.after(getDatasets)

    runNLP = dsl.ContainerOp(
        name="run NLP training",
        image="valorad/nlp_smart_dispatching:latest",

        pvolumes = {
            "/workspace/nlp-smart-dispatching/datasets": datasetVolume.volume,
            "/workspace/nlp-smart-dispatching/training/models": outputVolume.volume,
        }
    )

    runNLP.after(extractData)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(nlp_pipeline, __file__ + '.yaml')