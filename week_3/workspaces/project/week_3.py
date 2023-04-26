from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
    String
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(config_schema={"s3_key": String},
    out={"stocks": Out(dagster_type=List[Stock],
    description="List of stocks from S3")},
    required_resource_keys={"s3"},
    tags={"kind": "AWS S3"})
def get_s3_data(context):
    key_name = context.op_config["s3_key"]
    stocks = [Stock.from_list(x) for x in context.resources.s3.get_data(key_name)]
    return stocks


@op(ins={"stocks": In(dagster_type=List[Stock],
    description="List of stocks from S3")},
    out={"aggregation": Out(dagster_type=Aggregation, description="Stock with greatest high value")})
def process_data(stocks):
    max_high = max(stocks, key=lambda s: s.high)
    return Aggregation(date=max_high.date, high=max_high.high)


@op(ins={"aggregation": In(dagster_type=Aggregation,
    description="Stock with greatest high value")},
    required_resource_keys={"redis"},
    tags={"kind": "Redis"}
    )
def put_redis_data(context, aggregation):
    context.resources.redis.put_data(str(aggregation.date), str(aggregation.high))


@op(ins={"aggregation": In(dagster_type=Aggregation,
    description="Stock with greatest high value")},
    required_resource_keys={"s3"},
    tags={"kind": "AWS S3"}
    )
def put_s3_data(context, aggregation):
    key_name = f"aggregation_{aggregation.date}.json"
    context.resources.s3.put_data(key_name, aggregation)


@graph
def machine_learning_graph():
    max_aggregation = process_data(get_s3_data())
    put_redis_data(max_aggregation)
    put_s3_data(max_aggregation)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

@static_partitioned_config(partition_keys=[str(x) for x in range(1,11)])
def docker_config(partition_key: str):
    return {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key":"prefix/stock_{}.csv".format(partition_key)}}},
}


machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={
        "s3": mock_s3_resource,
        "redis": ResourceDefinition.mock_resource(),
    },
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker_config,
    resource_defs={
        "s3": s3_resource,
        "redis": redis_resource,
    },
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
)


machine_learning_schedule_local = ScheduleDefinition(job=machine_learning_job_local, cron_schedule="*/15 * * * *")


@schedule(cron_schedule="0 * * * *", job=machine_learning_job_docker)
def machine_learning_schedule_docker():
    pass


@sensor(job=machine_learning_job_docker, minimum_interval_seconds=30)
def machine_learning_sensor_docker(context):
    new_files = get_s3_keys("dagster", "prefix", "http://localstack:4566")
    if not new_files:
        yield SkipReason("No new s3 files found in bucket.")
        return
    for new_file in new_files:
        yield RunRequest(
            run_key=new_file,
            run_config={
                 "resources": {
                    "s3": {"config": S3},
                    "redis": {"config": REDIS},
                 },
                "ops": {
                    "get_s3_data": {"config": {"s3_key": new_file}},
                },
            },
        )
