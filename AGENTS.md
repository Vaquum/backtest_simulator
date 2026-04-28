## The project

We're not just building backtest. We're building market simulator and we get measured by backtest-to-live delta.

Our appoach is that backtest is pushed as close to paper trade as possible, and paper trade to live trade. backtest-vs-live and paper-vs-live are key metrics for us in addition to R and profit factor. Every review must go through this filter: if we water down something here, we water it down in paper and live as well. 

We can't game the system in that way, it will not work to our favour. 

`bts sweep` is the main pathway of this software. Humans only use `bts sweep`. Every task has to be reviewed under the question "which part of this can make bts sweep better, and did that part make bts better in this task". That delta should always be flat zero. If the sweep done by bts can be made better, that is what building this software is all about. IT's not about anything else.

## The Five Principles

- no duck-taping or saga ornamentation
- "bts or it didn't happen" -> everything is bts cli first
- not building a parallel universe around praxis/nexus ->  extending them
- honest to how market actuall works -> works just like nexus/praxis paper-trade
- strategy tested, is strategy deployed -> limen leads to nexus leads to praxis