"""Position-aware imitation diagnostics and training-class balancing."""
CLASS_NAMES = ('wait', 'enter_long', 'enter_short', 'keep_long', 'keep_short',
               'exit_long', 'exit_short')


def decision_classes(observations, votes, sides, threshold=.064):
    import torch
    if observations.ndim != 2 or observations.shape[1] < 11:
        raise ValueError('Missing fixed agent-state observation tail')
    held = observations[:, -11]
    if not torch.isfinite(held).all() or not torch.isfinite(votes).all():
        raise ValueError('Non-finite imitation decisions')
    if not ((held == -1) | (held == 0) | (held == 1)).all():
        raise ValueError('Position sign must be unscaled in the observation tail')
    result = torch.zeros_like(votes, dtype=torch.long)
    if 1 in sides:
        result[(held == 0) & (votes > threshold)] = 1
    if -1 in sides:
        result[(held == 0) & (votes < -threshold)] = 2
    result[held == 1] = 3
    result[held == -1] = 4
    result[(held == 1) & (votes < -threshold)] = 5
    result[(held == -1) & (votes > threshold)] = 6
    return result


def imitation_weights(observations, actions, sides):
    import torch
    labels = decision_classes(observations, actions[:, 0], sides)
    counts = torch.bincount(labels, minlength=len(CLASS_NAMES))
    present = (counts > 0).sum()
    if labels.numel() == 0:
        raise ValueError('Empty imitation dataset')
    return labels.numel() / (present * counts[labels].float())


def imitation_fidelity(observations, teacher_votes, predicted_votes, sides):
    target = decision_classes(observations, teacher_votes, sides)
    predicted = decision_classes(observations, predicted_votes, sides)
    classes = {}
    for index, name in enumerate(CLASS_NAMES):
        mask = target == index
        count = int(mask.sum().item())
        if count:
            classes[name] = {'count': count, 'recall': float((predicted[mask] == index).float().mean().item())}
    exits = (target == 5) | (target == 6)
    return {'macro_recall': sum(item['recall'] for item in classes.values()) / len(classes),
            'exit_count': int(exits.sum().item()),
            'exit_recall': float((predicted[exits] == target[exits]).float().mean().item()) if exits.any() else None,
            'classes': classes}
