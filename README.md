# allocation-policy

Implementation of allocation policies.

# Download

In the future I hope to upload this to pypi. For now pip install
directly from git.

```
> pip install git+https://github.com/HHRI-Transplantation/allocation-policy
```

# Usage

Currently only contains Acuity Circles.

```python
>>> from allocation_policy.acuity_circles import allocation_pathway
>>> don_non_hr_beat = 'Y'
>>> don_age_in_months = 507
>>> alloc_func = allocation_pathway(don_non_hr_beat, don_age_in_months)
>>> print(alloc_func)
<function policy_9_8_H at 0x000002A3E9512FB0>
```

Plug your polars frame into this function and it currently calculates the
tiers. Still need to implement sorting, tiebreaks, etc..
