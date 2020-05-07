# Meeting Notes

## 03/10/19

- Expand literature review in areas suggested by research proposal which work towards planning in multi agent systems
    - Consider verifying the published results in simulation
    - Focus on balance between formal methods and learning algorithms
        - Formal
            - POSG
            - POMDP
            - Swarm
        - Learning
            - MARL
- Work towards specifying a problem and building the relevant simulator
    - Consider the particular task (start formalising this)
    - Consider the restrictions/assumtions placed on the given method

- Smaller things to do/decide
    - Building a framework as a result of the PhD
    - Supervision/Co supervision
    - Particular problem to solve

## 10/10/19

- Get in touch with Antoine Cully
- Continue with literature review
    - Focus on breadth of available literature
    - Consider the verification of swarm intelligence
- Start building document to summarise findings

## 5/12/19

- Pad out the rest of the sections (such as Hard Coded/Markov Game) sections of the review
- Focus the research on looking at the intersection between Game-Theory, MARL and Model Predictive Control. To that end:
    - Continue an evaluation of the important techniques towards MARL problems
    - Search for literature which work at the intersection between MARL and MPC (perhaps using POMDPs as the predictive horizon etc)
    - Similar to the first point, continue an review in the important techniques and ideas in MPC, particularly looking at assumptions made, theoretical contributions and distributed settings
        - In terms of applications, let's focus in on those related to autonomous systems (or perhaps even looking at how the techniques can be applied to autonomous systems)
    - Re-evaluate the Swarm application for MPC. Perhaps it is not necessary/optimal.

## 7/05/20

- Establish equivalent of Eq (70) in Galla Supplementary Material for the coupled dynamics of Tuyls' Q-Learning

- Run tests on Tuyls' dynamics to determine the effect of the coupled term (from Tuyls et al Eq 11)

    $$\sum_{ij} x_i A_{ij}y_j.$$

    If removing this term has little consequence on the overall dynamics, then consider lifting the coupled term in derivation.

- Write up derivation so far with introduction and motivation

- (Added by Aamal) Begin running numerical tests on Q-Learning approach to determine islands of stability in an empirical test (i.e. produce equivalent figures as Figure 6 in Galla's paper)

- Continue working towards equivalent of Eq. 69 for Leung et al's mean field dynamics

- Keep in mind the choice of the second supervisor.


**Return to Literature Review - Revised**

- Populate some of the empty sections (such as Swarms and Stochastic Games)
- Add to the ideas in Section 7.1 (Considering Intelligence in Swarm Dynamics) with relevant literature and proposed research directions.
