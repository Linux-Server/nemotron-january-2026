#!/usr/bin/env python3
"""Terminate the benchmark EC2 instance (stops billing). SG + key pair are left for reuse.

  stt-benchmark/.venv/bin/python ec2-bench/ec2_down.py
"""
import json
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
st = json.loads((HERE / ".instance.json").read_text())
ec2 = boto3.Session(profile_name="419599258555_AWSAdministratorAccess").client("ec2", region_name=st["region"])
r = ec2.terminate_instances(InstanceIds=[st["instance_id"]])
state = r["TerminatingInstances"][0]["CurrentState"]["Name"]
print(f"terminate {st['instance_id']} ({st.get('itype')}) {st['region']} -> {state}")
print("(security group 'nemotron-bench-sg' + key pair left in place for reuse)")
