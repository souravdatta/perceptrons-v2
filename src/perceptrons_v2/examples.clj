(ns perceptrons-v2.examples
  (:use [perceptrons-v2.core]))

(def sdr-train-data
  [[1 1 0 1 1 1 1] ;; 0
   [0 1 0 1 0 0 0] ;; 1
   [1 1 1 0 1 1 0] ;; 2
   [1 1 1 1 1 0 0] ;; 3
   [0 1 1 1 0 0 1] ;; 4
   [1 0 1 1 1 0 1] ;; 5
   [1 0 1 1 1 1 1] ;; 6
   [1 1 1 1 0 0 0] ;; 7
   [1 1 1 1 1 1 1] ;; 8
   [1 1 1 1 1 0 1] ;; 9
   ])

(def sdr-labels
  [[1 0 0 0 0 0 0 0 0 0]
   [0 1 0 0 0 0 0 0 0 0]
   [0 0 1 0 0 0 0 0 0 0]
   [0 0 0 1 0 0 0 0 0 0]
   [0 0 0 0 1 0 0 0 0 0]
   [0 0 0 0 0 1 0 0 0 0]
   [0 0 0 0 0 0 1 0 0 0]
   [0 0 0 0 0 0 0 1 0 0]
   [0 0 0 0 0 0 0 0 1 0]
   [0 0 0 0 0 0 0 0 0 1]])

(def sdr-net (-> (network 7 [7 10] 0.5)
                 (train! sdr-train-data sdr-labels 6000)))

(predict sdr-net [1 0 1 1 1 0 1]) ;; predicts 5
(predict sdr-net [1 1 1 0 1 1 0]) ;; predicts 2

