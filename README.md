Overview

One of the questions that increasingly interests me is how we can represent complex real-world systems in a way that makes reasoning possible.

During a research project at the University of Texas Center for Space Research, I explored this question in an environmental context.

Suppose we are given an arbitrary location on Earth. Can we identify a protected ecosystem that is environmentally similar, not because it is geographically nearby, but because it shares similar climatic and ecological characteristics?

Answering that question requires more than geographic search. It requires defining what "similar" means, constructing a representation of environmental conditions, and building a retrieval system capable of searching that representation space.

To explore this problem, I developed an ecological analog discovery platform that maps locations into an environmental feature space and retrieves the most similar protected land from a nationwide database.

Framing the Problem

Environmental systems are high-dimensional.

Temperature, precipitation, elevation, vegetation dynamics, productivity, and water flux all interact to shape ecosystem behavior. Geographic distance alone is often a poor proxy for ecological similarity.

The core challenge therefore became:

How can environmental similarity be represented quantitatively?

Rather than treating locations as coordinates on a map, I represented them as vectors of environmental characteristics and searched for similarity within that representation.

Approach

The system constructs environmental representations using:

Long-term temperature averages
Long-term precipitation averages
Elevation

After standardizing these variables, locations are embedded into a shared feature space. Similarity search is then performed using nearest-neighbor retrieval to identify the most environmentally comparable protected area.

Once an analog is retrieved, the platform automatically gathers decades of satellite observations through Google Earth Engine and generates comparative analyses of:

Vegetation dynamics (NDVI)
Evapotranspiration
Land Surface Temperature
Gross Primary Productivity

This transforms a retrieval problem into a scientific hypothesis-generation tool. Users can move beyond identifying similar locations and begin evaluating whether those locations exhibit comparable ecological behavior over time.

Why This Project Matters To Me

What I found most interesting was that the project resembled a pattern that appears repeatedly throughout machine learning.

Many ML systems can be viewed as answering a simple question:

Given a representation of an object, how can we retrieve or reason about other objects that are meaningfully similar?

Recommendation systems, retrieval systems, scientific search, representation learning, and modern AI agents all rely on this idea.

This project explored that same principle in an environmental setting. The domain was ecology, but the underlying challenge was representation, retrieval, and reasoning under uncertainty.

Working on the project strengthened my interest in machine learning because it demonstrated how carefully chosen representations can transform a complex real-world problem into something computationally tractable.
