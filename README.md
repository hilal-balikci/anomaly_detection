# About The Dataset
The Numenta Anomaly Benchmark (NAB) is a novel benchmark for evaluating algorithms for anomaly detection in streaming, online applications. It is comprised of over 50 labeled real-world and artificial timeseries data files plus a novel scoring mechanism designed for real-time applications. All of the data and code is fully open-source, with extensive documentation, and a scoreboard of anomaly detection algorithms: github.com/numenta/NAB. The full dataset is included here, but please go to the repo for details on how to evaluate anomaly detection algorithms on NAB.

# NAB Data Corpus
The NAB corpus of 58 timeseries data files is designed to provide data for research in streaming anomaly detection. It is comprised of both real-world and artifical timeseries data containing labeled anomalous periods of behavior. Data are ordered, timestamped, single-valued metrics. All data files contain anomalies, unless otherwise noted.

The majority of the data is real-world from a variety of sources such as AWS server metrics, Twitter volume, advertisement clicking metrics, traffic data, and more. All data is included in the repository, with more details in the data readme. We are in the process of adding more data, and actively searching for more data. Please contact at nab@numenta.org if you have similar data (ideally with known anomalies) that you would like to see incorporated into NAB.

The NAB version will be updated whenever new data (and corresponding labels) is added to the corpus; NAB is currently in v1.0.



ambient_temperature_system_failure.csv: The ambient temperature in an office setting.

cpu_utilization_asg_misconfiguration.csv: From Amazon Web Services (AWS) monitoring CPU usage – i.e. average CPU usage across a given cluster. When usage is high, AWS spins up a new machine, and uses fewer machines when usage is low.

ec2_request_latency_system_failure.csv: CPU usage data from a server in Amazon's East Coast datacenter. The dataset ends with complete system failure resulting from a documented failure of AWS API servers. There's an interesting story behind this data in the Numenta blog.

machine_temperature_system_failure.csv: Temperature sensor data of aninternal component of a large, industrial mahcine. The first anomaly is aplanned shutdown of the machine. The second anomaly is difficult to detect and directly led to the third anomaly, a catastrophic failure of the machine.

nyc_taxi.csv: Number of NYC taxi passengers, where the five anomalies occur during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow storm. The raw data is from the NYC Taxi and Limousine Commission.

The data file included here consists of aggregating the total number of taxi passengers into 30 minute buckets. rogue_agent_key_hold.csv: Timing the key holds for several users of a computer, where the anomalies represent a change in the user.

rogue_agent_key_updown.csv: Timing the key strokes for several users of a computer, where the anomalies represent a change in the user. 

realTraffic/ : Real time traffic data from the Twin Cities Metro area in Minnesota, collected by the Minnesota Department of Transportation.

Included metrics include occupancy, speed, and travel time from specific sensors.

realTweets/ : A collection of Twitter mentions of large publicly-traded companies such as Google and IBM. The metric value represents the number of mentions for a given ticker symbol every 5 minutes.

# Artificial data
artificialNoAnomaly/ : Artifically-generated data without any anomalies.

artificialWithAnomaly/ : Artifically-generated data with varying types of anomalies.

# Acknowledgments
We encourage you to publish your results on running NAB, and share them with us at nab@numenta.org. Please cite the following publication when referring to NAB:

Lavin, Alexander and Ahmad, Subutai. "Evaluating Real-time Anomaly Detection Algorithms – the Numenta Anomaly Benchmark", Fourteenth International Conference on Machine Learning and Applications, December 2015.
