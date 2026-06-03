# 🌎 Protected Lands App

## Overview

For any scientific study, it is important to control for confounding variables. When evaluating whether protected status improves ecological outcomes, comparing two environmentally different locations can lead to misleading conclusions.

To address this challenge, our research sought to match any location to the most ecologically similar protected land based on climate and environmental characteristics. By identifying appropriate ecological analogs, we could create more meaningful comparisons when assessing whether protected areas exhibit statistically significant differences in indicators such as vegetation health, productivity, temperature, and evapotranspiration.

I developed the Protected Lands App to automate this process. The platform retrieves climate and elevation characteristics for a user-provided location, normalizes those variables using z-scores, and identifies the most similar protected land using nearest-neighbor search. It then automatically generates multi-year satellite-data comparisons to support downstream statistical analysis.

## Framing the Problem

The key question was:

> How can ecological similarity be represented quantitatively?

Environmental systems are inherently multidimensional. Temperature, precipitation, elevation, vegetation dynamics, ecosystem productivity, and water flux all contribute to ecosystem behavior.

Rather than treating locations as simple geographic coordinates, I represented them as vectors of environmental characteristics. The challenge then became identifying which locations occupied similar positions within this environmental feature space.

## Approach

The system first retrieves long-term climate and geographic characteristics for a user-provided location, including:

* Average annual temperature
* Annual precipitation
* Elevation

Because these variables exist on different scales, direct comparison would produce misleading results. To create a common representation, each feature is standardized using **z-score normalization**, transforming the variables into comparable units relative to their distributions.

After normalization, locations are embedded into a shared environmental feature space. Similarity is then computed using nearest-neighbor search, allowing the system to retrieve the protected land whose environmental characteristics most closely match the input location.

Once a match is identified, the platform automatically retrieves more than two decades of satellite observations through Google Earth Engine and generates comparative time-series analyses for:

* NDVI (vegetation health)
* Evapotranspiration (ET)
* Land Surface Temperature (LST)
* Gross Primary Productivity (GPP)

This enables researchers to move beyond simple geographic comparisons and evaluate whether environmentally similar locations exhibit statistically significant differences in ecological health indicators.

## Impact

* Automated ecological analog identification for environmental research workflows
* Enabled controlled comparisons between arbitrary locations and protected lands
* Integrated climate, elevation, and remote-sensing datasets into a unified analytical pipeline
* Generated multi-decade satellite-data comparisons through a single query
* Supported research conducted through the University of Texas Center for Space Research

## Why This Project Interested Me

What fascinated me most about this project was that it transformed a scientific question into a representation problem.

The challenge was not collecting more data. The challenge was determining how to represent environmental systems in a way that makes meaningful comparison possible.

That idea appears repeatedly throughout machine learning. Recommendation systems, retrieval systems, scientific search, and modern AI systems all depend on constructing useful representations and identifying meaningful similarity within them.

Although this project was developed in an environmental science context, it introduced me to many of the concepts that continue to draw me toward machine learning research: representation, retrieval, similarity, and using data-driven methods to reason about complex real-world systems.
