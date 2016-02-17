(ns cetiri.core
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [info endian-little]]]
            [vertigo
             [bytes :refer [direct-buffer byte-seq]]
             [structs :refer [wrap-byte-seq int8]]]))

(alter-var-root
  (var uncomplicate.clojurecl.core/*opencl-2*)
  (fn [f]  false))
(def data-file (io/file
                 (io/resource 
                   "examples/hello-kernel.cl" )))
"src/cetiri/core/hello-kernel.cl"
(let [notifications (chan)
      follow (register notifications)work-sizes (work-size [1])
      host-msg (direct-buffer 16)
      program-source
      "__kernel void hello_kernel(__global char16 *msg) {\n    *msg = (char16)('H', 'e', 'l', 'l', 'o', ' ',
   'k', 'e', 'r', 'n', 'e', 'l', '!', '!', '!', '\\0');\n}\n"
      ]
  ;(println "notifications: " notifications)
  ;(println  "follow: " follow)
  ;(println "work-sizes: " work-sizes)  
  ;(println "host-msg: " host-msg)  
  ;(println "program-source: " program-source)  
  
  (try
    (with-release [dev (first (devices (first (platforms))))
                  ctx (context [dev])
                   cqueue (command-queue ctx dev)
                   cl-msg (cl-buffer ctx 16 :write-only)
                   prog (build-program! (program-with-source ctx [program-source]))
                   hello-kernel (kernel prog "hello_kernel")
                   read-complete (event)
                   ]
      (println "dev1: " dev)
      (println "ctx1: " ctx)
      (println "cqueue1: " cqueue) 
      (println "prog1: " prog)
      (println "hello-kernel1: " hello-kernel)
      (println "read-complete1: " read-complete)

      (set-args! hello-kernel cl-msg)
      (enq-nd! cqueue  hello-kernel work-sizes)
      (enq-read! cqueue cl-msg host-msg)
      (apply str (map char
                      (wrap-byte-seq int8 (byte-seq host-msg))))
      
      
   #_(facts
     "Section 4.1, Page 69."
     (let [host-msg (direct-buffer 16)
           work-sizes (work-size [1])
           program-source
           (slurp (io/resource "examples/openclinaction/ch04/hello-kernel.cl"))]
      (println "host-msg: " host-msg)
      (println "work-sizes: " work-sizes)
      (println "program-source: " program-source)        
       
       
       (with-release [cl-msg (cl-buffer ctx 16 :write-only)
                      prog (build-program! (program-with-source ctx [program-source]))
                      hello-kernel (kernel prog "hello_kernel")
                      read-complete (event)]

         (set-args! hello-kernel cl-msg) => hello-kernel
         (enq-nd! cqueue hello-kernel work-sizes) => cqueue
         (enq-read! cqueue cl-msg host-msg read-complete) => cqueue
         (follow read-complete host-msg) => notifications
         (apply str (map char
                         (wrap-byte-seq int8 (byte-seq (:data (<!! notifications))))))
         => "Hello kernel!!!\0")))      
      
      
     )
    (catch Exception e (println "Greska 11111111: " (.getMessage e))))
  )

(println "drugi deo")

;(import 'java.net.URL)
;(def cnn (URL. "https://github.com/uncomplicate/clojurecl/blob/master/test/clojure/uncomplicate/clojurecl/examples/openclinaction/ch04.clj"))
;---------------------------------------------------------------------------------------
(let [notifications (chan)
      follow (register notifications)]
  
  ;(println "notifications: " notifications)
  ;(println "follow: " follow)

  ;(
    ;(println "konji")
    (try
     (with-release [dev (first (devices (first (platforms))))
                    ctx (context [dev])
                    cqueue (command-queue ctx dev)]
       ;)
     ;(catch Exception e (println "Greska: " (.getMessage e)))) 
    
      (println "dev: " dev)
      (println "ctx: " ctx)
      (println "cqueue: " cqueue)
      ;(.getHost cnn)       
    (facts
     "Section 4.1, Page 69."
     (let [host-msg (direct-buffer 16)
           work-sizes (work-size [1])        
           program-source 
           (slurp (io/resource "examples/hello-kernel.cl" ))
           ;"__kernel void hello_kernel(__global char16 *msg) {\n    *msg = (char16)('H', 'e', 'l', 'l', 'o', ' ',   'k', 'e', 'r', 'n', 'e', 'l', '!', '!', '!', '\\0');\n}\n"
           ]
       (println "program-source 2222: " program-source)       
       (with-release [cl-msg (cl-buffer ctx 16 :write-only)
                      prog (build-program! (program-with-source ctx [program-source]))
                      hello-kernel (kernel prog "hello_kernel")
                      read-complete (event)
                      ]
         
      (println "cl-msg: " cl-msg)
      (println "prog: " prog)
      (println "hello-kernel: " hello-kernel)         
      (println "read-complete: " read-complete) 
      
         (set-args! hello-kernel cl-msg) => hello-kernel
         (enq-nd! cqueue hello-kernel work-sizes) => cqueue
         (enq-read! cqueue cl-msg host-msg read-complete) => cqueue
         (follow read-complete host-msg) => notifications
         (apply str (map char
                         (wrap-byte-seq int8 (byte-seq (:data (<!! notifications))))))
         => "Hello kernel!!!\0")))

    (facts
     "Section 4.2, Page 72."
     (let [host-a (float-array [10])
           host-b (float-array [2])
           host-out (float-array 1)
           work-sizes (work-size [1])
           program-source
           (slurp (io/resource "examples/double-test.cl"))
           ;"#ifdef FP_64\n#pragma OPENCL EXTENSION cl_khr_fp64: enable\n#endif\n__kernel void double_test(__global float* a,\n                          __global float* b,\n                          __global float* out) {\n#ifdef FP_64\n    double c = (double)(*a / *b);\n    *out = (float)c;\n#else\n    *out = *a * *b;\n#endif\n}\n"
           ]
       (with-release [cl-a (cl-buffer ctx (* 2 Float/BYTES) :read-only)
                      cl-b (cl-buffer ctx (* 2 Float/BYTES) :read-only)
                      cl-out (cl-buffer ctx (* 2 Float/BYTES) :write-only)
                      prog (build-program! (program-with-source ctx [program-source])
                                           (if (contains? (info dev :extensions)
                                                          "cl_khr_fp64")
                                             "-DFP_64"
                                             "")
                                           notifications)
                      double-test (kernel prog "double_test")]

         (set-args! double-test cl-a cl-b cl-out) => double-test
         (enq-write! cqueue cl-a host-a) => cqueue
         (enq-write! cqueue cl-b host-b) => cqueue
         (enq-nd! cqueue double-test work-sizes) => cqueue
         (enq-read! cqueue cl-out host-out) => cqueue
         (seq host-out) => (map / host-a host-b))))

    (facts
     "Section 4.3, Page 77."
     (println "Single FP Config: " (info dev :single-fp-config)))

    (facts
     "Section 4.4.1, Page 79."
     (println "Preferred vector widths: "
              (select-keys (info dev) [:preferred-vector-width-char
                                       :preferred-vector-width-short
                                       :preferred-vector-width-int
                                       :preferred-vector-width-long
                                       :preferred-vector-width-float
                                       :preferred-vector-width-double
                                       :preferred-vector-width-long])))

    (facts
     "Section 4.4.4, Page 85."
     (let [host-data (byte-array 16)
           work-sizes (work-size [1])
           program-source
           (slurp (io/resource "examples/vector-bytes.cl"))
           ;"__kernel void vector_bytes(__global uchar16 *test) {\n    uint4 vec = (global uint4) (0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);\n    uchar *p = &vec;\n    *test = (uchar16)(*p, *(p+1), *(p+2), *(p+3), *(p+4), *(p+5), *(p+6),\n                      *(p+7), *(p+8), *(p+9), *(p+10), *(p+11), *(p+12),\n                     *(p+13), *(p+14), *(p+15));\n}\n"
           ]
       (with-release [cl-data (cl-buffer ctx 16 :write-only)
                      prog (build-program! (program-with-source ctx [program-source]))
                      vector-bytes (kernel prog "vector_bytes")]

         (set-args! vector-bytes cl-data) => vector-bytes
         (enq-write! cqueue cl-data host-data) => cqueue
         (enq-nd! cqueue vector-bytes work-sizes) => cqueue
         (enq-read! cqueue cl-data host-data) => cqueue
         (seq host-data) => (if (endian-little dev)
                              [3 2 1 0 7 6 5 4 11 10 9 8 15 14 13 12]
                              (range 16)))))
    
    
    
    ;)
  )
          (catch Exception e (println "Greska 222222: " (.getMessage e)))))
