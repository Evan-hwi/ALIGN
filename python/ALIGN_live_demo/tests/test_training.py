import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CHANNELS, load_model
from train_model import EXPECTED_FIELDS, evaluate, load_sessions, read_session, run


def csv_bytes(sid, subject, label, offset=0):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(EXPECTED_FIELDS)
    for seq in range(100):
        # Four independent recordings, identical format but distinct signals.
        vals = [.1 + .6 * label + offset + (seq % 7) * .0001] * 12
        writer.writerow([sid, subject, label, seq, 1000000 + seq * 20000] + vals)
    return stream.getvalue().encode('utf-8')


class TrainingTests(unittest.TestCase):
    def build_data(self, folder):
        data = Path(folder) / 'data'
        data.mkdir()
        for i, (subject, label) in enumerate([('P01', 0), ('P01', 1), ('P02', 0), ('P02', 1)]):
            (data / f'{i}.csv').write_bytes(csv_bytes(f'session{i}', subject, label, .005 * i))
        return data

    def test_train_to_saved_model_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build_data(tmp)
            out = Path(tmp) / 'model'
            args = argparse.Namespace(data=str(data), kind='body', split='session',
                                      out=str(out), overwrite=False)
            with redirect_stdout(io.StringIO()):
                model, evaluation = run(args)
            self.assertEqual(model['kind'], 'body')
            self.assertEqual(load_model(out / 'model.json')['training_windows'], 8)
            self.assertEqual(evaluation['windows'], 8)
            self.assertEqual(evaluation['correct'], 8)
            self.assertEqual(len(model['training_sessions']), 4)
            with self.assertRaises(ValueError):
                run(args)

    def test_both_split_modes_keep_groups_disjoint_and_test_each_session_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions = load_sessions(self.build_data(tmp))
            for split in ['session', 'subject']:
                with self.subTest(split=split):
                    report = evaluate(sessions, split)
                    tested = []
                    for fold in report['folds']:
                        self.assertFalse(set(fold['train_sessions']) & set(fold['test_sessions']))
                        if split == 'subject':
                            self.assertFalse(set(fold['train_subjects']) & set(fold['test_subjects']))
                        tested.extend(fold['test_sessions'])
                    self.assertEqual(sorted(tested), sorted(s['session_id'] for s in sessions))
                    self.assertEqual(report['windows'], 8)

    def test_copied_signal_with_new_id_cannot_leak_into_another_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build_data(tmp)
            (data / 'copy.csv').write_bytes(csv_bytes('new_session', 'P09', 0, 0))
            with self.assertRaisesRegex(ValueError, '중복'):
                load_sessions(data)

    def test_mixed_label_and_invalid_numeric_data_fail_explicitly(self):
        data = csv_bytes('s', 'P01', 0).decode('utf-8')
        mixed = data.replace('s,P01,0,99,', 's,P01,1,99,')
        bad = data.replace(',0.1,', ',nan,', 1)
        for value in [mixed, bad]:
            with self.subTest(data=value[-100:]), self.assertRaises(ValueError):
                read_session('bad.csv', value.encode('utf-8'))


def zero_imu2_csv_bytes(sid):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(EXPECTED_FIELDS)
    for seq in range(100):
        writer.writerow([sid, 'P01', 0, seq, 1000000 + seq * 20000] + [.1] * 6 + [0.0] * 6)
    return stream.getvalue().encode('utf-8')


class SessionRegistryTests(unittest.TestCase):
    BAD_SID = '20261113_180239_b5f86fa7'

    def build(self, folder, registry_rows=None):
        data = TrainingTests.build_data(None, folder)
        (data / f'P01_{self.BAD_SID}_label0.csv').write_bytes(zero_imu2_csv_bytes(self.BAD_SID))
        if registry_rows is not None:
            lines = ['session_id,date,subject,condition,label,mount,status,note']
            lines += [f'{sid},2026-11-13,P01,body,0,unrecorded,{status},{note}'
                      for sid, status, note in registry_rows]
            (Path(folder) / 'sessions.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return data

    def test_empty_sensor_session_is_rejected_without_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build(tmp)
            with self.assertRaisesRegex(ValueError, 'IMU2'):
                load_sessions(data, registry=None)

    def test_registry_in_parent_folder_excludes_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [(f'session{i}', 'ok', '') for i in range(4)]
            rows.append((self.BAD_SID, 'exclude', 'IMU2 all zero'))
            data = self.build(tmp, rows)
            report = {}
            sessions = load_sessions(data, report=report)
            self.assertEqual(len(sessions), 4)
            self.assertEqual(report['registry'], 'sessions.csv')
            self.assertEqual([x['session_id'] for x in report['excluded']], [self.BAD_SID])
            self.assertEqual(report['unregistered'], [])

    def test_check_and_unregistered_sessions_are_kept_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [('session0', 'check', 'loose strap'), ('session1', 'ok', ''),
                    (self.BAD_SID, 'exclude', 'IMU2 all zero')]
            data = self.build(tmp, rows)
            report = {}
            sessions = load_sessions(data, report=report)
            self.assertEqual(len(sessions), 4)
            self.assertEqual(report['check'], ['session0'])
            self.assertEqual(sorted(report['unregistered']), ['session2', 'session3'])

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build(tmp, [('session0', 'maybe', '')])
            with self.assertRaisesRegex(ValueError, 'status'):
                load_sessions(data)


if __name__ == '__main__':
    unittest.main()
