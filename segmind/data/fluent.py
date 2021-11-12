from segmind.data.public import upload as upload_data


def upload(path, destination_path, datastore_name):
    # we don't need to start a run, in order to upload data to Segmind cluster
    # run = _get_or_start_run()
    # run_id = run.info.run_id
    # experiment_id = run.info.experiment_id

    upload_data(
        path=path,
        datastore_name=datastore_name,
        destination_path=destination_path
    )
