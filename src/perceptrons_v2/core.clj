(ns perceptrons-v2.core
  (:use [perceptrons-v2.utils]))


(defn activation [x]
  (sigmoid x))

(defn perceptron [n]
  (atom {:input   nil
         :b-input nil
         :weights (random-weights (inc n))
         :output  nil
         :error   nil}))

(defn add-input! [p is]
  (do
    (swap! p
           merge {:input is :b-input (concat is [1.0])})
    p))

(defn add-weights! [p ws]
  (do
    (swap! p merge {:weights ws})
    p))

(defn run-perceptron! [p]
  (let [p-data @p]
    (swap! p
           merge
           {:output (activation
                      (dot-product
                        (:b-input p-data)
                        (:weights p-data)))})
    p))

(defn perceptron-output [p]
  (:output @p))

(defn network [n hidden-layers lr]
  {:state (atom {:layers (map-indexed
                           (fn [i layer-num]
                             (map
                               (fn [_]
                                 (perceptron (if (= i 0)
                                               n
                                               (nth hidden-layers
                                                    (dec i)))))
                               (range layer-num)))
                           hidden-layers)
                 :output nil})
   :rate  lr})

(defn output-with-error! [out-layer ys]
  (doseq [[p y] (map vector out-layer ys)]
    (let [p-data @p
          o (:output p-data)
          error (* o (- 1 o) (- y o))]
      (swap! p merge {:error error})))
  out-layer)

(defn hidden-layers-with-error! [hidden-layers out-with-errs]
  (let [hlayers (reverse hidden-layers)
        out-layers (concat (vector out-with-errs) (butlast hlayers))]
    (doseq [[hl ol] (map vector hlayers out-layers)]
      (doseq [[hp i] (map vector hl (range (count hl)))]
        (let [wsum (reduce + (map
                               (fn [p] (*
                                         (nth (:weights @p) i)
                                         (:error @p)))
                               ol))
              oh (perceptron-output hp)
              error (* oh (- 1 oh) wsum)]
          (swap! hp merge {:error error}))))
    hidden-layers))

(defn adjust-weights! [net]
  (doseq [layer (:layers @(:state net))]
    (doseq [p layer]
      (let [weights (:weights @p)
            inputs (:b-input @p)
            error (:error @p)
            new-weights (map (fn [w i] (+ w (* i error (:rate net))))
                             weights
                             inputs)]
        (swap! p merge {:weights new-weights})))))

(defn run-network! [net is]
  (let [state (:state net)
        layers (:layers @state)]
    (loop [ls layers
           input is]
      (if (empty? ls)
        (do
          (swap! state merge {:output input})
          net)
        (recur (rest ls)
               (do
                 (doseq [p (first ls)]
                   (-> p
                       (add-input! input)
                       (run-perceptron!)))
                 (map perceptron-output (first ls))))))))


(defn back-propagate! [net is labels]
  (let [state @(:state net)
        layers (:layers state)
        out-layer (last layers)
        hidden-layers (butlast layers)]
    (run-network! net is)
    (output-with-error! out-layer labels)
    (hidden-layers-with-error! hidden-layers out-layer)
    (adjust-weights! net)
    net))

(defn train! [net train-data train-labels epochs]
  (let [train-pair (map vector train-data train-labels)]
    (doseq [_ (range epochs)]
      (doseq [[is labels] train-pair]
        (back-propagate! net is labels)))
    net))

;(defn run-many [net n]
;  (doseq [_ (range n)]
;    (back-propagate! net [1 0] [1]))
;  net)

(defn predict [net is]
  (-> net
      (run-network! is)
      (:state)
      (deref)
      (:output)))

(comment

  (-> (perceptron 2)
      (add-weights! [10 10 -15])
      (add-input! [1 0]))

  ;; AND
  (-> (perceptron 2)
      (add-weights! [10 10 -15])
      (add-input! [1 0])
      (run-perceptron!))

  ;; OR
  (-> (perceptron 2)
      (add-weights! [10 10 -5])
      (add-input! [0 1])
      (run-perceptron!))

  ;; Network
  (network 2 [2 1] 0.5)

  ;; Network
  (network 7 [7 10] 0.5)

  ;(-> (network 2 [2 1] 0.5)
  ;    (run-many 30000))

  ;; Network run
  (-> (network 2 [2 1] 0.5)
      (back-propagate! [1 0] [1]))

  ;; Network train
  (-> (network 2 [2 1] 0.5)
      (train! [[1 0] [0 1] [1 1] [0 0]] [[1] [1] [0] [0]] 4000))

  ;; Network train and predict
  (def xor
    (-> (network 2 [2 1] 0.5)
        (train! [[1 0] [1 1] [0 1] [0 0]] [[1] [0] [1] [0]] 6000)))

  ;; Bigger networks
  (def xor2
    (-> (network 2 [2 2 1] 0.5)
        (train! [[1 0] [1 1] [0 1] [0 0]] [[1] [0] [1] [0]] 6000)))

  (predict xor2 [1 1])

  )