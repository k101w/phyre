#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque

import sortedcontainers


class Queue(object):
    def __init__(self):
        self._items = deque([])

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.popleft() \
            if not self.empty() else None

    def empty(self):
        return len(self._items) == 0

    def find(self, item):
        return self._items.index(item) if item in self._items else None
#TODO:
    def remove(self,item):
        self._items.remove(item)
    
    def compare_and_remove(self, j, node):
        if node < self._items[j]:
            self._items.pop(index=j)
            return True
        return False

class Stack(object):
    def __init__(self):
        self._items = list()

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop() if not self.empty() else None

    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._items)


class PriorityQueue(object):

    def __init__(self):
        self._queue = sortedcontainers.SortedList()

    def push(self, node):
        self._queue.add(node)

    def pop(self):
        return self._queue.pop(index=0)

    def empty(self):
        return len(self._queue) == 0

    def compare_and_replace(self, i, node):
        if node < self._queue[i]:
            self._queue.pop(index=i)
            self._queue.add(node)

    def find(self, node):
        try:
            loc = self._queue.index(node)
            return loc
        except ValueError:
            return None

class My_PriorityQueue(object):

    def __init__(self):
        self._queue = sortedcontainers.SortedList(key=lambda x:x.f)

    def push(self, node):
        self._queue.add(node)

    def pop(self):
        return self._queue.pop(index=0)

    def empty(self):
        return len(self._queue) == 0

    def compare_and_replace(self, i, node):
        if node.f < (self._queue[i]).f:
            self._queue.pop(index=i)
            self._queue.add(node)

    def find(self, node):
        try:
            loc = self._queue.index(node)
            return loc
        except ValueError:
            return None


class Set(object):
    def __init__(self):
        self._items = set()
#TODO:
    def add(self, items):
        for item in items:
            for i in item:
                self._items.add(i)

    def remove(self, item):
        self._items.remove(item)
#TODO:
    def find(self,item):
        return(item in self._items)


class Dict(object):
    def __init__(self):
        self._items = dict()

    def add(self, key, value):
        self._items.update({key: value})

    def remove(self, key):
        self._items.pop(key, None)

    def find(self, key):
        return self._items[key] if key in self._items else None


if __name__ == '__main__':
    a=PriorityQueue()
    a.push(1)
    a.push(2)
    b=My_PriorityQueue()
    b.push(1)
    b.push(2)
    print(b.find(1))
    
    print(a.pop())
    print(b.pop())