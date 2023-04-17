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


@op(
    description="Returns list of stock data from s3_key",
    config_schema={"s3_key": str},
    required_resource_keys={"s3"},
    out={"stocks": Out(dagster_type=List[Stock], description="List of stocks")},
)
def get_s3_data(context) -> List[Stock]:
    file_name = context.op_config["s3_key"]
    stocks = context.resources.s3.get_data(file_name)
    stocks = list(map(Stock.from_list, stocks))
    return stocks


@op(
    description="Returns the Aggregation from the stock value with the greatest high value",
    ins={"stocks": In(dagster_type=List[Stock], description="List of stocks")},
    out={
        "stock_greatest_high": Out(
            dagster_type=Aggregation,
            description="Date and high values from the stock value with the greatest high value",
        )
    },
)
def process_data(context, stocks) -> Aggregation:
    max_high_value = max(stocks, key=lambda x: x.high)
    return Aggregation(date=max_high_value.date, high=max_high_value.high)


@op(
    required_resource_keys={"redis"},
    description="Uploads stock_greatest_high (Aggregation class type) to Redis",
    ins={
        "stock_greatest_high": In(
            dagster_type=Aggregation,
            description="Date and high values from the stock value with the greatest high value",
        )
    },
)
def put_redis_data(context, stock_greatest_high):
    context.resources.redis.put_data(
        name=String(stock_greatest_high.date), value=String(stock_greatest_high.high)
    )


@op(
    required_resource_keys={"s3"},
    description="Upload stock_greatest_high (Aggregation class type) to S3 data lake",
    ins={
        "stock_greatest_high": In(
            dagster_type=Aggregation,
            description="Date and high values from the stock value with the greatest high value",
        )
    },
)
def put_s3_data(context, stock_greatest_high) -> Nothing:
    context.resources.s3.put_data(
        key_name=String(stock_greatest_high.date), data=stock_greatest_high
    )


@graph
def machine_learning_graph():
    data = get_s3_data()
    highest_data = process_data(data)
    put_redis_data(highest_data)
    put_s3_data(highest_data)


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
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
)