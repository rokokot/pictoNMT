""" Statistics calculation utility """

import time
import math
import sys

from eole.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(
        self,
        loss=0,
        auxloss=0,
        n_batchs=0,
        n_sents=0,
        n_tokens=0,
        n_correct=0,
        computed_metrics=None,
        data_stats=None,
    ):
        self.loss = loss
        self.auxloss = auxloss
        self.n_batchs = n_batchs
        self.n_sents = n_sents
        self.n_tokens = n_tokens
        self.n_correct = n_correct
        self.n_src_tokens = 0
        self.computed_metrics = computed_metrics if computed_metrics is not None else {}
        self.data_stats = data_stats if data_stats is not None else {}
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from eole.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_tokens=True)
        return our_stats

    def update(self, stat, update_n_src_tokens=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_tokens(bool): whether to update (sum) `n_src_tokens`
                or not

        """
        self.loss += stat.loss
        self.auxloss += stat.auxloss
        self.n_batchs += stat.n_batchs
        self.n_sents += stat.n_sents
        self.n_tokens += stat.n_tokens
        self.n_correct += stat.n_correct
        self.computed_metrics = stat.computed_metrics
        for cid in stat.data_stats.keys():
            if cid in self.data_stats.keys():
                self.data_stats[cid]["count"] += stat.data_stats[cid]["count"]
            else:
                self.data_stats[cid] = {}
                self.data_stats[cid]["count"] = stat.data_stats[cid]["count"]
            self.data_stats[cid]["index"] = stat.data_stats[cid]["index"]

        if update_n_src_tokens:
            self.n_src_tokens += stat.n_src_tokens

    def computed_metric(self, metric):
        """check if metric(TER/BLEU) is computed and return it"""
        assert metric in self.computed_metrics, "Metric {} not found".format(metric)
        return self.computed_metrics[metric]

    def accuracy(self):
        """compute accuracy"""
        return 100 * (self.n_correct / self.n_tokens)

    def xent(self):
        """compute cross entropy"""
        return self.loss / self.n_tokens

    def aux_loss(self):
        return self.auxloss / self.n_sents

    def ppl(self):
        """compute perplexity"""
        return math.exp(min(self.loss / self.n_tokens, 100))

    def elapsed_time(self):
        """compute elapsed time"""
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            (
                "Step %s; acc: %2.1f; ppl: %5.2f; xent: %2.2f; aux: %2.3f; "
                + "lr: %7.2e; sents: %7.0f; bsz: %4.0f/%4.0f/%2.0f; "
                + "%3.0f/%3.0f tok/s; %6.0f sec;"
            )
            % (
                step_fmt,
                self.accuracy(),
                self.ppl(),
                self.xent(),
                self.aux_loss(),
                learning_rate,
                self.n_sents,
                self.n_src_tokens / self.n_batchs,
                self.n_tokens / self.n_batchs,
                self.n_sents / self.n_batchs,
                self.n_src_tokens / (t + 1e-5),
                self.n_tokens / (t + 1e-5),
                time.time() - start,
            )
            + "".join([" {}: {}".format(k, round(v, 2)) for k, v in self.computed_metrics.items()])
        )
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, patience, step):
        """display statistics to tensorboard"""
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        for k, v in self.computed_metrics.items():
            writer.add_scalar(prefix + "/" + k, round(v, 4), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_tokens / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
        if patience is not None:
            writer.add_scalar(prefix + "/patience", patience, step)
