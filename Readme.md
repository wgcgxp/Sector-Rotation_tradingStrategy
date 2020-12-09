### Environment
python 3.6\
requirement.txt

### Usage
These files is used for measuring the prosperity of industries in the market.
- The sector rotation is indicated by a **complex score**, which is made of scores calculated according to *financial reports*, *performance forecast*, *performance express*, and the *consensus expectation* of the some stocks. The complex score can give trading signals for each industry.

### Output
- The complex indicator and the signals given by each selected indicators.
- The performance of the investment in a certain period by using the complex indicator
- Some in-between files according to needs.

### Content of the Project
- Codes
	- Base
		- Some reusable functions
		- Variables - stores the datasets processed by different scripts
	- Raw data porcessing
		- PPI data: financial report data
		- IPI data: performance forecast & express data
		- predict data: consensus expectation data
	- Data calculation
		- Weight dataset
		- Dataset merge, unify names and fill in Null values
		- Weighted close index
		- Calculate weighted scores of all industries within a period of time
		- Change the score into long short indicators
		- Construct a trading strategy according to the indicators
		- Measure the performance of the indicators
		- Construct a complex indicator and measure the performance
	- Rename
		- rename to columns, select columns [Enum]
- Data\
A seperate file, not uploaded.
