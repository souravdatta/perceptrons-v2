(ns perceptrons-v2.utils)

(def rnd (java.util.Random.))

(defn random-weights [n]
  (map (fn [_] (. rnd nextDouble 0 1)) (range n)))

(defn dot-product [xs ys]
  (reduce + (map * xs ys)))

(defn assoc-seq [s index v]
  (map-indexed (fn [i x] (if (= i index) v x)) s))

(defn sigmoid [x]
  (/ 1.0 (+ 1 (Math/exp (- x)))))