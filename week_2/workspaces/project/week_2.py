from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
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
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
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
    config=docker,
    resource_defs={
        "s3": s3_resource,
        "redis": redis_resource,
    },
)
